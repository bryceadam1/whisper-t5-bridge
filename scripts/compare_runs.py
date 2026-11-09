"""Markdown comparison of evaluated runs: overall agreement per condition, and per family for one condition.

  python -m scripts.compare_runs --runs base_newprotocol A_lexical B_encoder_input AB C_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = {"eval_heldout/truncated": "held-out voices, truncated", "eval_seen/truncated": "seen voices, truncated",
              "eval_heldout/full": "held-out voices, full 30 s", "eval_seen/full": "seen voices, full 30 s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--family-condition", default="eval_heldout/truncated")
    args = ap.parse_args()

    runs = {r: json.load(open(Path(args.results) / r / "summary.json")) for r in args.runs}
    labels = dict(zip(args.runs, args.labels or args.runs))
    ref = runs[args.runs[0]]
    print("| run | " + " | ".join(f"{v}: EM / agree / KL" for v in CONDITIONS.values()) + " |")
    print("|---|" + "---|" * len(CONDITIONS))
    for r, s in runs.items():
        cells = [f"{s[c]['bridge']['ALL']['exact']:.3f} / {s[c]['bridge']['ALL']['token_agree']:.3f} / "
                 f"{s[c]['bridge']['ALL']['kl']:.3f}" for c in CONDITIONS]
        print(f"| {labels[r]} | " + " | ".join(cells) + " |")
    for system, name in (("asr_control", "ASR control"), ("instr_only", "instruction only")):
        cells = [f"{ref[c][system]['ALL']['exact']:.3f} / {ref[c][system]['ALL']['token_agree']:.3f} / "
                 f"{ref[c][system]['ALL']['kl']:.3f}" for c in CONDITIONS]
        print(f"| *{name}* | " + " | ".join(cells) + " |")

    c = args.family_condition
    fams = sorted(ref[c]["bridge"], key=lambda f: (f == "ALL", f))
    print(f"\n**{CONDITIONS[c]}**, exact match / mean KL by family\n")
    print("| family | n | " + " | ".join(labels[r] for r in runs) + " | ASR control |")
    print("|---|---|" + "---|" * (len(runs) + 1))
    for f in fams:
        cells = [f"{s[c]['bridge'][f]['exact']:.2f} / {s[c]['bridge'][f]['kl']:.2f}" for s in runs.values()]
        a = ref[c]["asr_control"][f]
        print(f"| {f} | {ref[c]['bridge'][f]['n']} | " + " | ".join(cells) + f" | {a['exact']:.2f} / {a['kl']:.2f} |")


if __name__ == "__main__":
    main()
