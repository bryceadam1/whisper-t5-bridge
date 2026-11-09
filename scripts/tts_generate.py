"""Synthesise the spoken inputs with Edge0/Audio8-TTS-Realistic-Air (runs on the cloud GPU only).

The model needs trust_remote_code=True. Its code was read before use and the revision is pinned below;
its codec.pth is loaded by that code with torch.load(weights_only=True).

Each input is voiced by cloning a LibriSpeech reference clip (see select_voices.py):
  train items      -> one rendering by a training speaker          (split "train")
  eval items       -> one by a training speaker                    (split "eval_seen")
                      and one by a held-out speaker                (split "eval_heldout")
The reference clip is chosen among the speaker's slow/medium/fast clips so the expected duration stays
under ~21 s, then a WSOLA tempo change in [0.9, 1.15] varies the rate further. Output is 16 kHz mono
FLAC plus a manifest line per clip. Re-running resumes: finished ids are skipped.

  python -m scripts.tts_generate --split train --limit 20000
  python -m scripts.tts_generate --benchmark           # throughput at several batch sizes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ID = "Edge0/Audio8-TTS-Realistic-Air"
MODEL_REVISION = "23c305cc013fde378edb459f538317043c993a74"
FRAMES_PER_SECOND = 44100 / 2048
MAX_SECONDS = 25.0
LENGTH_MARGIN = 1.8  # generation budget: this many times the longest expected raw duration in a batch, + 2 s
TARGET_SECONDS = 21.0
TEMPO_RANGE = (0.9, 1.15)
SAMPLING = dict(temperature=0.8, top_p=0.95, top_k=50, do_sample=True)


def stable_rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def batch_seconds(batch) -> float:
    raw = max(j["est_seconds"] * j["tempo"] for j in batch)
    return min(MAX_SECONDS / TEMPO_RANGE[0], LENGTH_MARGIN * raw + 2.0)


def plan(items: list[dict], voices: list[dict], split: str) -> list[dict]:
    """Assign (speaker, reference clip, tempo) to each item for the given split."""
    pool = "heldout" if split == "eval_heldout" else "train"
    by_speaker: dict[str, list[dict]] = {}
    for v in voices:
        if v["split"] == pool:
            by_speaker.setdefault(v["speaker"], []).append(v)
    speakers = sorted(by_speaker)
    jobs = []
    for item in items:
        rng = stable_rng(item["id"], split)
        speaker = rng.choice(speakers)
        words = len(item["tts_text"].split())
        tempo = rng.uniform(*TEMPO_RANGE)
        fits = [v for v in by_speaker[speaker] if words / v["words_per_sec"] / tempo <= TARGET_SECONDS]
        ref = rng.choice(fits) if fits else max(by_speaker[speaker], key=lambda v: v["words_per_sec"])
        if not fits:
            tempo = TEMPO_RANGE[1]
        jobs.append({"id": item["id"], "split": split, "text": item["tts_text"], "speaker": speaker,
                     "ref_clip": ref["clip"], "ref_text": ref["text"], "tempo": round(tempo, 3),
                     "est_seconds": words / ref["words_per_sec"] / tempo})
    return jobs


class TTS:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
                                               dtype=torch.bfloat16).eval().to(device)
        self.sample_rate = self.model.config.codec_sample_rate
        self.ref_codes: dict[str, np.ndarray] = {}

    def reference_codes(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self.ref_codes:
            audio, sr = sf.read(key, dtype="float32")
            import torchaudio.functional as AF

            wav = AF.resample(self.torch.from_numpy(audio), sr, self.sample_rate).to(self.device)
            codes, lengths = self.model.encode_audio(wav[None, None], self.torch.tensor([wav.numel()], device=self.device))
            self.ref_codes[key] = codes[0, :, : int(lengths[0])].cpu().numpy()
        return self.ref_codes[key]

    def generate(self, texts, ref_paths, ref_texts, max_seconds):
        torch = self.torch
        codes = [self.reference_codes(p) for p in ref_paths]
        inputs = self.processor(text=texts, reference_codes=codes, reference_text=ref_texts, return_tensors="pt")
        max_new = int(max_seconds * FRAMES_PER_SECOND) + 8
        # The model sizes its KV cache, and attends over it every step, at config.max_seq_len (2048).
        # Both read the config at call time, so size it to this batch's prompt plus output instead.
        width = int((inputs["prefix_attention_mask"].sum(1) + inputs["reference_code_lengths"]
                     + inputs["suffix_attention_mask"].sum(1)).max())
        self.model.config.max_seq_len = min(2048, width + max_new + 1)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, return_dict_in_generate=True, **SAMPLING)
            wavs, lengths = self.model.decode_audio(out.codes)
        import torchaudio.functional as AF

        wavs = AF.resample(wavs.float(), self.sample_rate, 16000)
        lengths = (lengths.float() * 16000 / self.sample_rate).long()
        return [(wavs[i, : int(lengths[i])].cpu().numpy(), bool(out.finished[i])) for i in range(len(texts))]


class FakeTTS:
    """Stand-in for local pipeline tests: a tone whose length follows the text."""
    def generate(self, texts, ref_paths, ref_texts, max_seconds):
        out = []
        for t in texts:
            n = int(16000 * min(max_seconds, 0.4 * len(t.split()) + 0.5))
            out.append((0.1 * np.sin(np.arange(n) * 2 * np.pi * 220 / 16000).astype(np.float32), True))
        return out


def apply_tempo(audio: np.ndarray, tempo: float) -> np.ndarray:
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter

    writer = ArrayWriter(1)
    wsola(1, speed=tempo).run(ArrayReader(audio[None].astype(np.float64)), writer)
    return writer.data[0].astype(np.float32)


def finish(job, audio, finished, out_dir: Path):
    audio = apply_tempo(audio, job["tempo"])
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    path = out_dir / job["split"] / f"{job['id']}.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 16000)
    rec = {k: job[k] for k in ("id", "split", "speaker", "ref_clip", "tempo")}
    rec.update(path=str(path.relative_to(out_dir)), duration=round(len(audio) / 16000, 3),
               finished=finished, too_long=len(audio) / 16000 > MAX_SECONDS)
    return rec


def run(jobs, tts, out_dir: Path, voices_dir: Path, batch_size: int, max_minutes: float | None = None,
        chunk: int = 512):
    manifest = out_dir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["id"] + "|" + json.loads(l)["split"] for l in manifest.open()}
    jobs = [j for j in jobs if j["id"] + "|" + j["split"] not in done]
    print(f"{len(jobs)} clips to generate ({len(done)} already done)", flush=True)
    lock, pool = threading.Lock(), ThreadPoolExecutor(4)
    t0, audio_s, n = time.time(), 0.0, 0

    def write(job, audio, finished):
        rec = finish(job, audio, finished, out_dir)
        with lock, manifest.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    for c in range(0, len(jobs), chunk):
        if max_minutes is not None and (time.time() - t0) / 60 > max_minutes:
            print(f"time limit of {max_minutes} min reached; stopping cleanly", flush=True)
            break
        # Within a chunk (still in priority order overall), batch similar lengths together.
        part = sorted(jobs[c:c + chunk], key=lambda j: j["est_seconds"])
        for b in range(0, len(part), batch_size):
            batch = part[b:b + batch_size]
            results = tts.generate([j["text"] for j in batch], [voices_dir / j["ref_clip"] for j in batch],
                                   [j["ref_text"] for j in batch], batch_seconds(batch))
            for job, (audio, finished) in zip(batch, results):
                audio_s += len(audio) / 16000
                pool.submit(write, job, audio, finished)
            n += len(batch)
        el = time.time() - t0
        print(f"{n}/{len(jobs)} clips, {audio_s / 3600:.2f} h audio, {audio_s / el:.1f}x realtime, "
              f"{el / 60:.1f} min", flush=True)
    pool.shutdown(wait=True)


def benchmark(jobs, tts, voices_dir: Path, sizes=(16, 32, 64, 128)):
    rng = random.Random(0)
    sample = rng.sample(jobs, 512)
    for bs in sizes:
        # as in run(): a batch of neighbours in expected length
        batch = sorted(sample, key=lambda j: j["est_seconds"])[256 - bs // 2: 256 + (bs + 1) // 2]
        tts.generate([batch[0]["text"]], [voices_dir / batch[0]["ref_clip"]], [batch[0]["ref_text"]], 5)  # warm
        t = time.time()
        res = tts.generate([j["text"] for j in batch], [voices_dir / j["ref_clip"] for j in batch],
                           [j["ref_text"] for j in batch], batch_seconds(batch))
        el = time.time() - t
        secs = sum(len(a) for a, _ in res) / 16000
        print(f"batch {bs:3}: {secs:7.1f} s audio in {el:6.1f} s -> {secs / el:6.1f}x realtime, "
              f"unfinished {sum(not f for _, f in res)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/flan_subset")
    ap.add_argument("--voices", default="data/voices")
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--split", choices=["train", "eval_seen", "eval_heldout"], default="train")
    ap.add_argument("--limit", type=int, default=None, help="first N items in priority order")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-minutes", type=float, default=None, help="stop cleanly between chunks after this")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--fake", action="store_true", help="local pipeline test without the TTS model")
    args = ap.parse_args()

    src = "train" if args.split == "train" else "eval"
    items = [json.loads(l) for l in open(Path(args.subset) / f"{src}.jsonl")][: args.limit]
    voices = json.loads((Path(args.voices) / "voices.json").read_text())
    jobs = plan(items, voices, args.split)
    tts = FakeTTS() if args.fake else TTS()
    if args.benchmark:
        benchmark(jobs, tts, Path(args.voices))
    else:
        run(jobs, tts, Path(args.out), Path(args.voices), args.batch_size, args.max_minutes)


if __name__ == "__main__":
    main()
