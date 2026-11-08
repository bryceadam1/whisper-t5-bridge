"""Pick reference voices for TTS cloning from LibriSpeech (CC BY 4.0, public-domain LibriVox readings).

Training voices come from dev-clean, held-out voices from test-clean, balanced by sex. Each speaker
gets three 6-12 s reference clips at their slowest, median and fastest speaking rate, so cloned speech
inherits a spread of rates. Writes <out>/voices.json and copies the clips to <out>/clips/.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import soundfile as sf


def read_speakers(root: Path) -> dict[str, dict]:
    speakers = {}
    for line in (root / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";"):
            continue
        sid, sex, subset, _minutes, name = [x.strip() for x in line.split("|", 4)]
        speakers[sid] = {"sex": sex, "subset": subset, "name": name}
    return speakers


def sentence_case(text: str) -> str:
    words = text.lower().split()
    words = ["I" if w == "i" else ("I" + w[1:] if w.startswith("i'") else w) for w in words]
    out = " ".join(words)
    return out[:1].upper() + out[1:] + "."


def utterances(root: Path, subset: str, sid: str):
    for trans in sorted((root / subset / sid).glob("*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            info = sf.info(str(path))
            dur = info.frames / info.samplerate
            if 6.0 <= dur <= 12.0:
                yield {"utt": uid, "path": path, "text": text, "duration": dur,
                       "rate": len(text.split()) / dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", default="data/raw/librispeech/LibriSpeech")
    ap.add_argument("--out", default="data/voices")
    ap.add_argument("--train", type=int, default=32)
    ap.add_argument("--heldout", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.librispeech), Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    speakers = read_speakers(root)
    rng = random.Random(args.seed)
    voices = []
    for split, subset, n in (("train", "dev-clean", args.train), ("heldout", "test-clean", args.heldout)):
        present = sorted(p.name for p in (root / subset).iterdir() if p.is_dir())
        for sex in ("F", "M"):
            pool = [s for s in present if speakers[s]["sex"] == sex]
            rng.shuffle(pool)
            chosen = 0
            for sid in pool:
                utts = sorted(utterances(root, subset, sid), key=lambda u: u["rate"])
                if len(utts) < 3 or chosen == n // 2:
                    continue
                chosen += 1
                for label, u in zip(("slow", "medium", "fast"), (utts[0], utts[len(utts) // 2], utts[-1])):
                    dest = out / "clips" / f"{sid}_{label}.flac"
                    shutil.copyfile(u["path"], dest)
                    voices.append({
                        "speaker": sid, "sex": sex, "split": split, "subset": subset, "rate_label": label,
                        "clip": str(dest.relative_to(out)), "text": sentence_case(u["text"]),
                        "duration": round(u["duration"], 2), "words_per_sec": round(u["rate"], 2),
                        "source_utt": u["utt"],
                    })
            assert chosen == n // 2, f"only {chosen} {sex} speakers in {subset}"
    (out / "voices.json").write_text(json.dumps(voices, indent=1))
    shutil.copyfile(root / "LICENSE.TXT", out / "LIBRISPEECH_LICENSE.TXT")
    rates = [v["words_per_sec"] for v in voices]
    print(f"{len({v['speaker'] for v in voices})} speakers, {len(voices)} clips, "
          f"rate {min(rates):.2f}-{max(rates):.2f} words/s")


if __name__ == "__main__":
    main()
