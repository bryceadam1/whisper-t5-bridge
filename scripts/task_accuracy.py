"""Task accuracy against the datasets' own answers (not agreement with the teacher), per task family.

Gold answers are recovered from the FLAN 2021 rows the eval inputs came from (data/flan_subset/eval_gold.json).
Scoring per task:
  sst2, trec            label accuracy (an answer given as an option letter is mapped to the option text)
  arc                   accuracy: the answer (an option label, or free text mapped to the option it overlaps
                        most) must be the correct option
  openbookqa            not scored: its eval prompts are open-ended (no options), so there is no fixed answer
  trivia_qa, nat. q.    lenient exact match: normalised gold contained in the answer, or equal to it
  lambada               the answer's first word equals the missing word
  common_gen            concept coverage: share of the given concepts that appear in the output
Scored for every system in a results/<run>/rows.json, and for the teacher (flan-t5-small on the true text).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

FAMILY = {"sst2": "sentiment", "trec": "question_type", "trivia_qa": "closed_book_qa",
          "natural_questions": "closed_book_qa", "arc": "multiple_choice",
          "lambada": "continuation", "common_gen": "concept_to_sentence"}
LABEL = re.compile(r"^\W*(?:[a-h]|[ivx]{1,4}|\d)\W*$", re.I)


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def option_text(pred: str, options: str | None) -> str:
    """If pred is just an option label like '(B)', '[ii]' or '2.', return that option's text."""
    if not options or not LABEL.match(pred.strip()):
        return pred
    core = re.sub(r"\W", "", pred).lower()
    for m in re.finditer(r"[\(\[]?\b([a-h]|[ivx]{1,4}|\d)\b[\)\]]?\.?\s+([^;\n]+)", options, re.I):
        if m.group(1).lower() == core:
            return re.split(r"\s+[\(\[]?(?:[a-h]|[ivx]{1,4}|\d)[\)\]]?\.\s", m.group(2))[0]
    return pred


def parse_options(options: str) -> list[str]:
    body = re.sub(r"^\s*(options|opt|choose from|choose your answer from|pick from)\s*(are)?\s*:?", "", options, flags=re.I)
    parts = re.split(r"\n|;|\s\+\s|^\+\s|\s-|^-|\[[ivx]+\]\s|\([a-h]\)\.?\s|\b[a-h]\)\.?\s|\b\d\.\s", body, flags=re.I)
    return [p.strip(" .-+") for p in parts if p and p.strip(" .-+")]


def choose(pred: str, options: str | None) -> str:
    """Map an answer to one of the options: by label position, else by largest word overlap."""
    opts = parse_options(options) if options else []
    if not opts:
        return pred
    core = re.sub(r"\W", "", pred).lower()
    if LABEL.match(pred.strip()):
        idx = ("abcdefgh".find(core) if len(core) == 1 and core.isalpha() and core not in "ivx" else
               ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"].index(core) if core in ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"] else
               int(core) - 1 if core.isdigit() else -1)
        return opts[idx] if 0 <= idx < len(opts) else pred
    words = set(norm(pred).split())
    best = max(opts, key=lambda o: len(words & set(norm(o).split())) / (len(set(norm(o).split())) or 1))
    return best if words & set(norm(best).split()) else pred


def score(task: str, pred: str, golds: list[str], item: dict) -> float:
    golds = [g for g in golds if not LABEL.match(g)] or golds
    p = norm(option_text(pred, item.get("options")))
    if task == "common_gen":
        concepts = [c.strip() for c in item["input"].split(",") if c.strip()]
        words = p.split()
        return sum(any(w.startswith(norm(c)[:4]) for w in words) for c in concepts) / max(1, len(concepts))
    if task == "lambada":
        return float(bool(p) and p.split()[0] == norm(golds[0]))
    if task in ("sst2", "trec"):
        return float(any(norm(g).split()[0] in p.split() for g in golds if norm(g)))
    if task == "arc":
        c = norm(choose(pred, item.get("options")))
        return float(any(c == norm(g) for g in golds))
    return float(any(norm(g) and (norm(g) in p or p == norm(g)) for g in golds))


def build_gold(parquet: str, out: str):
    """Recover each eval item's FLAN target(s) by matching zero-shot rows back to the eval inputs."""
    import pyarrow.parquet as pq

    from whisper_t5_bridge import flan

    templates = flan.load_templates(Path("data/raw/flan_templates"))
    by_input = {json.loads(l)["input"]: json.loads(l)["id"] for l in open("data/flan_subset/eval.jsonl")}
    gold = collections.defaultdict(set)
    f = pq.ParquetFile(parquet)
    for rg in range(f.num_row_groups):
        for row in f.read_row_group(rg, columns=["inputs", "targets", "task_name", "template_type"]).to_pylist():
            spec = flan.task_spec(row["task_name"]) if row["template_type"].startswith("zs") else None
            m = spec and flan.match(row["inputs"], templates[spec[0]])
            if m and m[0] in by_input:
                gold[by_input[m[0]]].add(row["targets"].strip())
    json.dump({k: sorted(v) for k, v in gold.items()}, open(out, "w"))
    print(f"gold answers for {len(gold)} of {len(by_input)} eval items -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-gold", metavar="PARQUET", help="recover gold answers from a FLAN 2021 shard first")
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--condition", default="eval_heldout/truncated")
    args = ap.parse_args()
    if args.build_gold:
        build_gold(args.build_gold, "data/flan_subset/eval_gold.json")
    if not args.runs:
        return

    items = {json.loads(l)["id"]: json.loads(l) for l in open("data/flan_subset/eval.jsonl")}
    gold = json.load(open("data/flan_subset/eval_gold.json"))
    labels = dict(zip(args.runs, args.labels or args.runs))
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        rows = json.load(open(Path("results") / run / "rows.json"))[args.condition]
        for r in rows:
            name = labels[run] if r["system"] == "bridge" else r["system"]
            if r["system"] != "bridge" and run != args.runs[0]:
                continue  # controls are identical across runs; take them from the first
            if r["task"] not in FAMILY:
                continue
            fam = FAMILY[r["task"]]
            table[name][fam].append(score(r["task"], r["output"], gold[r["id"]], items[r["id"]]))
            if r["system"] == "bridge" and run == args.runs[0]:
                table["teacher (text)"][fam].append(score(r["task"], r["teacher"], gold[r["id"]], items[r["id"]]))
    fams = sorted({f for t in table.values() for f in t})
    mean = lambda name, f: sum(table[name][f]) / len(table[name][f])
    # Families the teacher actually solves (beats the instruction-only floor by >= 0.2) can separate systems.
    informative = [f for f in fams if mean("teacher (text)", f) - mean("instr_only", f) >= 0.2]
    order = ["teacher (text)", "asr_control"] + [labels[r] for r in args.runs] + ["instr_only"]
    print(f"Task scores vs dataset answers ({args.condition}); macro = mean over families\n")
    print("| system | " + " | ".join(fams) + " | macro | **informative macro** | % of teacher |")
    print("|---|" + "---|" * (len(fams) + 3))
    teacher_inf = sum(mean("teacher (text)", f) for f in informative) / len(informative)
    for name in order:
        if name not in table:
            continue
        vals = [mean(name, f) for f in fams]
        inf = sum(mean(name, f) for f in informative) / len(informative)
        print(f"| {name} | " + " | ".join(f"{v:.2f}" for v in vals) +
              f" | {sum(vals) / len(vals):.3f} | **{inf:.3f}** | {100 * inf / teacher_inf:.0f}% |")
    print(f"\ninformative families (teacher beats instruction-only by >= 0.2): {', '.join(informative)}")


if __name__ == "__main__":
    main()
