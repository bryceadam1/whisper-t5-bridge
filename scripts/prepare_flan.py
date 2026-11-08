"""Select the FLAN subset: speakable single-input prompts from permissively licensed tasks.

Reads a DataProvenanceInitiative/flan2021_submix_original parquet shard, keeps zero-shot rows of the
tasks in whisper_t5_bridge.flan.TASKS, recovers each row's input field by matching FLAN's templates,
filters for short speakable English, deduplicates, and writes:

  <out>/train.jsonl  one line per unique input, ordered so that any prefix is balanced across tasks
  <out>/eval.jsonl   held-out inputs (disjoint from train), one rendering each

Each line carries up to K renderings (template, full prompt, instruction = prompt minus the input).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

from whisper_t5_bridge import flan

TRAIN_CAPS = {
    "sst2": 4000, "trec": 1800, "trivia_qa": 4500, "natural_questions": 4500, "common_gen": 3500,
    "arc": 1500, "openbookqa": 400, "lambada": 1500,
}
EVAL_PER_TASK = 120
MAX_WORDS = 50  # ~20 s at a normal rate, < 25 s after a 0.9x tempo change; enforced again on audio
MIN_WORDS = 2
SPEAKABLE = re.compile(r"[A-Za-z0-9 .,;:'\"!?()&%$-]+")
# A backstop against explicit or spam text that slips into otherwise curated sources; clinical or
# literary words ("sexual reproduction", "The Naked Ape") are kept.
EXPLICIT = re.compile(
    r"\b(porn\w*|xhamster|xvideos|hclips|txxx|xxx\w*|tits|boobs?|milfs?|cfnm|blowjobs?|cumshots?|cocks?|"
    r"pussy|fuck\w*|dildos?|sluts?|whores?|hentai|anal|penetration|camgirls?|escorts?|bdsm|fetish\w*|nsfw|"
    r"viagra|cialis|gangbang|handjobs?|orgasm\w*|horny|bitch\w*|stripper|onlyfans)\b", re.I)
SPAM = re.compile(r"\b\d+ (minutes?|hours?|days?|weeks?|months?|years?) ago\b|\bago \d{2}:?\d{2}\b", re.I)


def speech_text(text: str) -> str:
    """Light detokenisation for TTS (sst2 is PTB-tokenised: "it 's" , "do n't", "word ,")."""
    text = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    text = re.sub(r" ([.,;:!?%)])", r"\1", text)
    text = re.sub(r"([($]) ", r"\1", text)
    text = text.replace("`` ", '"').replace(" ''", '"').replace("-lrb-", "(").replace("-rrb-", ")")
    return re.sub(r"\s+", " ", text).strip()


def speakable(text: str) -> bool:
    words = text.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS or len(text) > 320:
        return False
    if not SPEAKABLE.fullmatch(text) or "http" in text or "www." in text:
        return False
    if EXPLICIT.search(text) or SPAM.search(text):
        return False
    digits = sum(c.isdigit() for c in text)
    return digits <= 0.1 * len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", default="data/flan_subset")
    ap.add_argument("--templates-cache", default="data/raw/flan_templates")
    ap.add_argument("--renderings", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    templates = flan.load_templates(Path(args.templates_cache))
    pool: dict[str, dict[str, dict]] = collections.defaultdict(dict)  # task -> norm input -> item
    stats = collections.Counter()
    for path in args.parquet:
        f = pq.ParquetFile(path)
        for rg in range(f.num_row_groups):
            for row in f.read_row_group(rg).to_pylist():
                if not row["template_type"].startswith("zs"):
                    continue
                spec = flan.task_spec(row["task_name"])
                if spec is None:
                    continue
                key, _, family = spec
                stats[f"{key}/rows"] += 1
                m = flan.match(row["inputs"], templates[key])
                if m is None:
                    continue
                value, options = m
                tts = speech_text(value)
                if not speakable(tts):
                    continue
                norm = re.sub(r"\W+", " ", value.lower()).strip()
                item = pool[key].get(norm)
                if item is None:
                    pool[key][norm] = {
                        "task": key, "source_task": row["task_name"], "family": family,
                        "input": value, "tts_text": tts, "options": options,
                    }
                elif options and not item["options"]:
                    item["options"] = options

    rng = random.Random(args.seed)
    train, evals = [], []
    for key, items in sorted(pool.items()):
        items = sorted(items.values(), key=lambda x: x["input"])
        rng.shuffle(items)
        n_eval = min(EVAL_PER_TASK, len(items) // 5)
        for split, chunk, k in (("eval", items[:n_eval], 1),
                                ("train", items[n_eval:n_eval + TRAIN_CAPS[key]], args.renderings)):
            for i, item in enumerate(chunk):
                item["id"] = f"{key}-{hashlib.sha1(item['input'].encode()).hexdigest()[:10]}"
                item["renderings"] = [
                    {"template": t, "prompt": p, "instruction": ins}
                    for t, p, ins in flan.sample_renderings(templates[key], item["input"], item["options"], k, rng)
                ]
                if split == "train":
                    item["priority"] = (i + rng.random()) / TRAIN_CAPS[key]
                    train.append(item)
                else:
                    evals.append(item)
        stats[f"{key}/unique"] = len(items)
        stats[f"{key}/eval"] = n_eval
        stats[f"{key}/train"] = min(TRAIN_CAPS[key], len(items) - n_eval)

    train.sort(key=lambda x: x["priority"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", evals)):
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
    for key in sorted(pool):
        print(f"{key:18} rows {stats[key + '/rows']:6}  unique {stats[key + '/unique']:6}  "
              f"train {stats[key + '/train']:5}  eval {stats[key + '/eval']:4}")
    words = [len(r["tts_text"].split()) for r in train]
    print(f"train {len(train)} inputs, {sum(len(r['renderings']) for r in train)} pairs, "
          f"mean {sum(words) / len(words):.1f} words; eval {len(evals)}")


if __name__ == "__main__":
    main()
