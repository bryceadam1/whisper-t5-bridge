"""Turn generated FLAN-style tasks (gen_tasks_local.py output) into a subset file for teacher_cache.py and
train.py: one item per transcript, its tasks as renderings, the transcript as the input.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from whisper_t5_bridge.flan import tidy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--transcripts", required=True, help="JSONL with id and transcript (the clip's input text)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = {json.loads(l)["id"]: json.loads(l)["transcript"] for l in open(args.transcripts)}
    by_id = collections.defaultdict(list)
    for line in open(args.tasks):
        t = json.loads(line)
        before, _, after = t["template"].partition("{x}")
        by_id[t["id"]].append({"template": t["template"], "family": t["family"], "gold": t["gold"],
                               "prompt": (before + text[t["id"]] + after).strip(), "instruction": tidy(before + after)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as fh:
        for i, renderings in by_id.items():
            fh.write(json.dumps({"id": i, "task": "libri_flan", "family": "libri_flan", "input": text[i],
                                 "tts_text": text[i], "options": None, "renderings": renderings}) + "\n")
    fams = collections.Counter(r["family"] for rs in by_id.values() for r in rs)
    print(f"{len(by_id)} transcripts, {sum(fams.values())} tasks: {dict(fams)}")


if __name__ == "__main__":
    main()
