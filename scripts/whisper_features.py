"""Whisper pass over the synthesised clips: quality check, encoder states, and control transcripts.

For each clip of a split:
  * transcript from standard 30 s Whisper decoding (batched) and its WER against the TTS text; clips
    with WER > --max-wer or longer than 25 s are marked keep=false
  * truncated-window encoder states (audio + 3 s silence, even frame count), float16
  * eval splits only: full 30 s window states, and a transcript decoded from the truncated-window states
    (the text-only control, matching what the game's Whisper would see)

Encoder states for clips of identical length are batched together, so padding never alters them.
Writes <out>/<split>_<window>.npy (all frames concatenated), <out>/<split>_meta.jsonl (offsets etc.).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch

from whisper_t5_bridge.audio import HOP, LogMel, SAMPLE_RATE, SILENCE_SECONDS, encode, window_samples
from whisper_t5_bridge.student import load_whisper, whisper_transcribe

MAX_SECONDS = 25.0
LEAD_IN = (0.1, 0.4)  # seconds of silence prepended: the TTS starts speech almost at sample 0, which
                      # clips onsets for Whisper; real push-to-talk audio has some lead-in anyway


def with_lead_in(audio: np.ndarray, clip_id: str) -> np.ndarray:
    seconds = random.Random(clip_id).uniform(*LEAD_IN)
    return np.concatenate([np.zeros(int(seconds * SAMPLE_RATE), np.float32), audio])


def enc_frames(n_samples: int, window: str) -> int:
    if window == "full":
        return 1500
    n = n_samples + int(SILENCE_SECONDS * SAMPLE_RATE)
    return -(-n // (2 * HOP))  # mel frames / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/audio")
    ap.add_argument("--subset", default="data/flan_subset")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--split", required=True, help="train, eval_seen, eval_heldout, or a libri* split")
    ap.add_argument("--skip-qc", action="store_true",
                    help="real recordings with ground-truth transcripts: keep every clip within the length limit")
    ap.add_argument("--max-wer", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    fe, wtok, whisper = load_whisper(device, dtype)
    logmel = LogMel(fe.mel_filters).to(device)
    src = "eval" if args.split.startswith("eval") else "train"
    text_of = {json.loads(l)["id"]: json.loads(l)["tts_text"] for l in open(Path(args.subset) / f"{src}.jsonl")}
    clips = [json.loads(l) for l in open(Path(args.audio) / "manifest.jsonl")]
    clips = [c for c in clips if c["split"] == args.split]
    seen = set()
    clips = [c for c in clips if not (c["id"] in seen or seen.add(c["id"]))]
    windows = ["truncated"] + (["full"] if args.split.startswith("eval") else [])
    print(f"{len(clips)} clips, windows {windows}", flush=True)

    audio = {c["id"]: with_lead_in(sf.read(str(Path(args.audio) / c["path"]), dtype="float32")[0], c["id"])
             for c in clips}
    meta = {c["id"]: {"id": c["id"], "speaker": c["speaker"], "duration": round(len(audio[c["id"]]) / SAMPLE_RATE, 3),
                      "finished": c["finished"]} for c in clips}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) Quality check: standard 30 s window transcription, batched.
    ids = [c["id"] for c in clips]
    if args.skip_qc:
        for m in meta.values():
            m.update(transcript_full=None, wer=0.0, keep=bool(m["finished"] and m["duration"] <= MAX_SECONDS))
    for b in range(0, 0 if args.skip_qc else len(ids), args.batch_size):
        chunk = ids[b:b + args.batch_size]
        feats = torch.stack([logmel(torch.from_numpy(window_samples(audio[i], "full")).to(device)) for i in chunk])
        with torch.no_grad():
            gen = whisper.generate(input_features=feats.to(dtype), max_new_tokens=160)
        for i, text in zip(chunk, wtok.batch_decode(gen, skip_special_tokens=True)):
            ref, hyp = wtok.normalize(text_of[i]), wtok.normalize(text.strip())
            wer = jiwer.wer(ref, hyp) if ref.strip() else 1.0
            m = meta[i]
            m.update(transcript_full=text.strip(), wer=round(wer, 4))
            m["keep"] = bool(m["finished"] and m["duration"] <= MAX_SECONDS and wer <= args.max_wer)
    print(f"QC done in {time.time() - t0:.0f} s; keep {sum(m['keep'] for m in meta.values())}/{len(meta)}", flush=True)

    # 2) Encoder states, batched over clips of identical length.
    for window in windows:
        lengths = {i: enc_frames(len(audio[i]), window) for i in ids}
        offsets, total = {}, 0
        for i in ids:
            offsets[i], total = total, total + lengths[i]
        arr = np.lib.format.open_memmap(out / f"{args.split}_{window}.npy", mode="w+", dtype=np.float16,
                                        shape=(total, 512))
        groups = collections.defaultdict(list)
        for i in ids:
            groups[lengths[i]].append(i)
        for n_frames, group in groups.items():
            for b in range(0, len(group), args.batch_size):
                chunk = group[b:b + args.batch_size]
                mel = torch.stack([logmel(torch.from_numpy(window_samples(audio[i], window)).to(device)) for i in chunk])
                states = encode(whisper.model.encoder, mel.to(dtype))
                assert states.shape[1] == n_frames, (states.shape, n_frames)
                for i, s in zip(chunk, states):
                    arr[offsets[i]:offsets[i] + n_frames] = s.float().cpu().numpy().astype(np.float16)
                    if window == "truncated" and args.split.startswith("eval"):
                        meta[i]["transcript_truncated"] = whisper_transcribe(whisper, wtok, s[None])
        arr.flush()
        for i in ids:
            meta[i][f"{window}_offset"], meta[i][f"{window}_frames"] = offsets[i], lengths[i]
        print(f"{window} states: {total} frames ({total * 1024 / 1e9:.2f} GB) in {time.time() - t0:.0f} s", flush=True)

    with open(out / f"{args.split}_meta.jsonl", "w") as fh:
        for i in ids:
            fh.write(json.dumps(meta[i]) + "\n")
    wers = [m["wer"] for m in meta.values()]
    print(f"done in {time.time() - t0:.0f} s; mean WER {np.mean(wers):.3f}, median {np.median(wers):.3f}")


if __name__ == "__main__":
    main()
