"""Apply FLAN's own sentence-level templates to speech transcripts (round 3 data).

Any transcript can be wrapped in these FLAN 2021 templates, which flan-t5-small was trained on:
  sentiment     sst2 templates that do not mention movies, with FLAN-style answer options
  keywords      common_gen's inverted templates ("What are the keywords in the following sentence: ...");
                the teacher copies exact words, a lexical signal
  continuation  lambada's completion templates
Each transcript gets --renderings templates from different families. Output is a subset file for
teacher_cache.py / train.py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from whisper_t5_bridge import flan

SENTIMENT_OPTIONS = ["Options are: (A). negative; (B). positive;", "Choose your answer from: 1. negative. 2. positive.",
                     "Options:\n[a]. negative;\n[b]. positive;"]


def families(cache_dir: Path) -> dict[str, list[str]]:
    lits = [flan._patterns_literal(p) for p in flan._download(cache_dir)]
    out = {"sentiment": [], "keywords": [], "continuation": []}
    for key, field, fam in (("sst2", "sentence", "sentiment"), ("common_gen", "target", "keywords"),
                            ("lambada", "sentence", "continuation")):
        for patterns in lits:
            for prompt, _ in patterns.get(key, []):
                fields = set(re.findall(r"\{(\w+)\}", prompt))
                if field not in fields or not fields <= {field, "options_"}:
                    continue
                if fam == "sentiment" and ("movie" in prompt.lower() or "review" in prompt.lower()):
                    continue
                if fam == "continuation" and (prompt.strip() == "{sentence}" or "_" in prompt):
                    continue
                t = prompt.replace("{" + field + "}", "{x}")
                if t not in out[fam]:
                    out[fam].append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="existing subset with id and input (the transcript)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--templates-cache", default="data/raw/flan_templates")
    ap.add_argument("--renderings", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fams = families(Path(args.templates_cache))
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "train.jsonl", "w") as fh:
        for line in open(Path(args.subset) / "train.jsonl"):
            item = json.loads(line)
            text, renderings = item["input"], []
            for fam in rng.sample(sorted(fams), args.renderings):
                t = rng.choice(fams[fam]).replace("{options_}", rng.choice(SENTIMENT_OPTIONS))
                before, _, after = t.partition("{x}")
                renderings.append({"template": t, "family": fam, "prompt": (before + text + after).strip(),
                                   "instruction": flan.tidy(before + after)})
            fh.write(json.dumps({"id": item["id"], "task": "speech_flan_templates", "family": "speech_flan_templates",
                                 "input": text, "tts_text": text, "options": None, "renderings": renderings}) + "\n")
            n += 1
    print(f"{n} transcripts x {args.renderings} renderings; templates per family: "
          f"{ {k: len(v) for k, v in fams.items()} }")


if __name__ == "__main__":
    main()
