"""Cache the teacher: flan-t5-small greedy outputs and top-k log-probs on the full text prompts.

For every rendering of every item: tokenised instruction (for the student), teacher output ids, and the
teacher's top-k log-probabilities (raw logits, temperature 1) at each output position.
Writes <out>/<split>.pt as a list of records.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from whisper_t5_bridge.student import MAX_NEW_TOKENS, load_t5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/flan_subset")
    ap.add_argument("--out", default="data/teacher")
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, t5 = load_t5(device)
    items = [json.loads(l) for l in open(Path(args.subset) / f"{args.split}.jsonl")][: args.limit]
    flat = [(it, r, x) for it in items for r, x in enumerate(it["renderings"])]
    flat.sort(key=lambda f: len(f[2]["prompt"]))
    records, t0 = [], time.time()
    for b in range(0, len(flat), args.batch_size):
        batch = flat[b:b + args.batch_size]
        enc = tok([x["prompt"] for _, _, x in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = t5.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                              output_logits=True, return_dict_in_generate=True)
        seqs = out.sequences[:, 1:]
        logits = torch.stack(out.logits, dim=1).float()  # (B, L, V)
        logp = torch.log_softmax(logits, -1)
        top_lp, top_id = logp.topk(args.topk, dim=-1)
        for i, (it, r, x) in enumerate(batch):
            ids = seqs[i].tolist()
            n = ids.index(tok.eos_token_id) + 1 if tok.eos_token_id in ids else len(ids)
            records.append({
                "id": it["id"], "rendering": r, "task": it["task"], "family": it["family"], "input": it["input"],
                "instruction_ids": tok(x["instruction"], truncation=True, max_length=512).input_ids,
                "prompt": x["prompt"], "template": x["template"],
                "out_ids": torch.tensor(ids[:n], dtype=torch.int32),
                "topk_ids": top_id[i, :n].to(torch.int32).cpu(),
                "topk_logp": top_lp[i, :n].to(torch.float16).cpu(),
                "text": tok.decode(ids[:n], skip_special_tokens=True),
                "finished": tok.eos_token_id in ids,
            })
        if (b // args.batch_size) % 20 == 0:
            print(f"{len(records)}/{len(flat)} in {time.time() - t0:.0f} s", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save(records, Path(args.out) / f"{args.split}.pt")
    lens = [len(r["out_ids"]) for r in records]
    print(f"{len(records)} records, mean output {sum(lens) / len(lens):.1f} tokens, "
          f"unfinished {sum(not r['finished'] for r in records)}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
