"""Local pipeline smoke test only: voice a few items with macOS `say` instead of the real TTS model.

Writes the same manifest format as tts_generate.py so the downstream scripts can be exercised on a
laptop. This audio is never used for training or reported results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

SEEN = ["Samantha", "Daniel", "Fred", "Eddy (English (US))", "Flo (English (UK))", "Reed (English (US))"]
HELDOUT = ["Karen", "Moira", "Rishi"]


def say(text: str, voice: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        wav = Path(d) / "x.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32", str(aiff), str(wav)], check=True)
        audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    return AF.resample(torch.from_numpy(audio), sr, 16000).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/flan_subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--eval", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    jobs = []
    train = [json.loads(l) for l in open(Path(args.subset) / "train.jsonl")][: args.train]
    evals = [json.loads(l) for l in open(Path(args.subset) / "eval.jsonl")][: args.eval]
    for i, it in enumerate(train):
        jobs.append((it, "train", SEEN[i % len(SEEN)]))
    for i, it in enumerate(evals):
        jobs.append((it, "eval_seen", SEEN[i % len(SEEN)]))
        jobs.append((it, "eval_heldout", HELDOUT[i % len(HELDOUT)]))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w") as fh:
        for it, split, voice in jobs:
            audio = say(it["tts_text"], voice)
            path = out / split / f"{it['id']}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, 16000)
            fh.write(json.dumps({"id": it["id"], "split": split, "speaker": voice, "ref_clip": None, "tempo": 1.0,
                                 "path": str(path.relative_to(out)), "duration": round(len(audio) / 16000, 3),
                                 "finished": True, "too_long": len(audio) / 16000 > 25}) + "\n")
    print(f"{len(jobs)} clips")


if __name__ == "__main__":
    main()
