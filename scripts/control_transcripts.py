"""Recompute the text-only control's Whisper transcripts for an eval split, in place in its meta file.

Both windows are decoded with the same openai/whisper-style decoder (token suppression, temperature
fallback on repetition), from the same lead-in-padded audio the features were computed on. Runs fine
on a CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jiwer
import soundfile as sf
import torch

from scripts.whisper_features import with_lead_in
from whisper_t5_bridge.audio import LogMel, encode, window_samples
from whisper_t5_bridge.student import load_whisper, whisper_transcribe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/audio")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--subset", default="data/flan_subset")
    ap.add_argument("--split", choices=["eval_seen", "eval_heldout"], required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe, tok, whisper = load_whisper(device)
    logmel = LogMel(fe.mel_filters).to(device)
    text = {json.loads(l)["id"]: json.loads(l)["tts_text"] for l in open(Path(args.subset) / "eval.jsonl")}
    paths = {json.loads(l)["id"]: json.loads(l)["path"] for l in open(Path(args.audio) / "manifest.jsonl")
             if json.loads(l)["split"] == args.split}
    meta_path = Path(args.features) / f"{args.split}_meta.jsonl"
    meta = [json.loads(l) for l in open(meta_path)]
    t0 = time.time()
    for k, m in enumerate(meta):
        audio = with_lead_in(sf.read(str(Path(args.audio) / paths[m["id"]]), dtype="float32")[0], m["id"])
        for window in ("truncated", "full"):
            mel = logmel(torch.from_numpy(window_samples(audio, window)).to(device))
            m[f"transcript_{window}"] = whisper_transcribe(whisper, tok, encode(whisper.model.encoder, mel[None]))
        if k % 200 == 0:
            print(f"{k}/{len(meta)} {time.time() - t0:.0f} s", flush=True)
    with open(meta_path, "w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    keep = [m for m in meta if m["keep"]]
    for window in ("truncated", "full"):
        wer = jiwer.wer([tok.normalize(text[m["id"]]) for m in keep], [tok.normalize(m[f"transcript_{window}"]) for m in keep])
        print(f"{args.split} {window}: corpus WER {wer:.3f} on {len(keep)} kept clips")


if __name__ == "__main__":
    main()
