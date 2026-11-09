"""Real speech for alignment: LibriSpeech (CC BY 4.0) utterances as extra training data (experiment C).

Writes, in the same formats the rest of the pipeline reads:
  <subset>/train.jsonl     one item per utterance: transcript as input, 2 renderings from generic templates
  <audio>/manifest.jsonl   one clip per utterance (split "libri"), paths relative to the LibriSpeech root

The templates are task-agnostic so every utterance can carry them. Translation is the most useful: the
teacher's output then depends on every word, a dense lexical signal. Utterances over 20 s are skipped
(25 s Whisper budget, and long translations would overrun the teacher's output limit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf

from whisper_t5_bridge.flan import tidy

TEMPLATES = [
    ("Translate to German: {x}", "translation"),
    ("Translate to French: {x}", "translation"),
    ("Translate the following sentence to German:\n{x}", "translation"),
    ("{x}\n\nSummarize the text above in a few words.", "summary"),
    ("{x}\n\nWhat is the main topic of this text?", "topic"),
    ("Write a question about the following text.\n\n{x}", "question_generation"),
    ("Continue the following story: {x}", "continuation"),
]
MAX_SECONDS = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LibriSpeech root containing train-clean-100/")
    ap.add_argument("--subsets", nargs="+", default=["train-clean-100"])
    ap.add_argument("--subset-out", default="data/libri_subset")
    ap.add_argument("--renderings", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    items, clips = [], []
    rng = random.Random(args.seed)
    for subset in args.subsets:
        for trans in sorted((root / subset).glob("*/*/*.trans.txt")):
            for line in trans.read_text().splitlines():
                uid, text = line.split(" ", 1)
                path = trans.parent / f"{uid}.flac"
                info = sf.info(str(path))
                dur = info.frames / info.samplerate
                if dur > MAX_SECONDS:
                    continue
                text = text.lower()
                cid = f"libri-{hashlib.sha1(uid.encode()).hexdigest()[:10]}"
                renderings = []
                for template, family in rng.sample(TEMPLATES, args.renderings):
                    before, _, after = template.partition("{x}")
                    renderings.append({"template": template, "prompt": (before + text + after).strip(),
                                       "instruction": tidy(before + after), "family": family})
                items.append({"id": cid, "task": "librispeech", "family": "librispeech", "input": text,
                              "tts_text": text, "options": None, "source_utt": uid, "renderings": renderings})
                clips.append({"id": cid, "split": "libri", "speaker": uid.split("-")[0], "ref_clip": None,
                              "tempo": 1.0, "path": str(path.relative_to(root)), "duration": round(dur, 3),
                              "finished": True, "too_long": False})
    out = Path(args.subset_out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    with open(root / "manifest.jsonl", "w") as fh:
        for c in clips:
            fh.write(json.dumps(c) + "\n")
    hours = sum(c["duration"] for c in clips) / 3600
    print(f"{len(items)} utterances ({hours:.1f} h) from {len({c['speaker'] for c in clips})} speakers, "
          f"{sum(len(i['renderings']) for i in items)} renderings")


if __name__ == "__main__":
    main()
