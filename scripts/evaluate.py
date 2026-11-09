"""Evaluate student-teacher agreement on held-out FLAN examples, by task family.

Systems, each compared with the teacher (flan-t5-small on the true text prompt):
  bridge         instruction via T5's encoder + spoken input via Whisper encoder -> bridge
  asr_control    the bridge bypassed: Whisper's own transcript (decoded from the same window) put in
                 place of the input in the text prompt
  instr_only     floor: the decoder sees the instruction alone, no input at all
Metrics: exact match of greedy outputs, token agreement and mean KL(teacher || system) per token with
teacher forcing along the teacher's output (full vocabulary, teacher run live).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
from pathlib import Path

import torch

from whisper_t5_bridge.bridge import load_bridge
from whisper_t5_bridge.data import Examples, length_batches, to_device
from whisper_t5_bridge.student import (add_cross_attn_lora, decoder_logits, encode_text, greedy, kl_full, load_t5,
                                       lora_disabled, student_encoder_states)
from safetensors.torch import load_file


def load_lora(t5, path):
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        rank = int(f.metadata()["rank"])
    adapters = add_cross_attn_lora(t5, rank=rank)
    state = load_file(path)
    for name, m in adapters.items():
        m.A.data.copy_(state[f"{name}.lora_A"])
        m.B.data.copy_(state[f"{name}.lora_B"])
    return adapters


def text_states(tok, t5, texts, device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    return encode_text(t5, enc.input_ids, enc.attention_mask), enc.attention_mask.bool()


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, t5 = load_t5(device)
    bridge = load_bridge(args.bridge).to(device).eval() if args.bridge else None
    adapters = load_lora(t5, args.lora) if args.lora else {}
    results = {}
    for split in args.splits:
        for window in args.windows:
            ex = Examples(args.features, args.teacher, split, tok, args.subset, window=window)
            rows = []
            for idxs in length_batches(ex, args.batch_size, shuffle=False):
                b = to_device(ex.collate(idxs), device)
                recs = b["records"]
                labels = b["labels"]
                valid = labels >= 0
                t_states, t_mask = text_states(tok, t5, [r["prompt"] for r in recs], device)
                with lora_disabled(adapters):
                    teacher_logits = decoder_logits(t5, t_states, t_mask, labels)
                # name -> (encoder states, mask, decoder uses the LoRA adapters)
                systems = {}
                if bridge is not None:
                    systems["bridge"] = (*student_encoder_states(t5, bridge, b), True)
                key = "transcript_truncated" if window == "truncated" else "transcript_full"
                prompts = [r["prompt"].replace(r["input"], ex.meta[r["id"]][key], 1) for r in recs]
                systems["asr_control"] = (*text_states(tok, t5, prompts, device), False)
                systems["instr_only"] = (encode_text(t5, b["instr_ids"], b["instr_mask"]), b["instr_mask"].bool(), False)
                if adapters:
                    # How far the adapters move the plain text path, if they were merged into the decoder.
                    systems["text_with_lora"] = (t_states, t_mask, True)
                for name, (states, mask, use_lora) in systems.items():
                    with contextlib.nullcontext() if use_lora else lora_disabled(adapters):
                        logits = decoder_logits(t5, states, mask, labels)
                        gen = tok.batch_decode(greedy(t5, states, mask), skip_special_tokens=True)
                    kl = kl_full(logits, teacher_logits, valid)
                    agree = (logits.argmax(-1) == labels) & valid
                    for j, r in enumerate(recs):
                        n = int(valid[j].sum())
                        rows.append({"system": name, "id": r["id"], "family": r["family"], "task": r["task"],
                                     "exact": gen[j].strip() == r["text"].strip(), "agree": int(agree[j].sum()),
                                     "kl": float(kl[j].sum()), "tokens": n, "output": gen[j], "teacher": r["text"]})
            results[f"{split}/{window}"] = rows
            print(f"{split}/{window}: {len(ex)} examples", flush=True)
    return results


def summarise(results):
    table = {}
    for cond, rows in results.items():
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r["system"], r["family"])].append(r)
            groups[(r["system"], "ALL")].append(r)
        for (system, family), rs in groups.items():
            tok = sum(r["tokens"] for r in rs)
            table.setdefault(cond, {}).setdefault(system, {})[family] = {
                "n": len(rs), "exact": sum(r["exact"] for r in rs) / len(rs),
                "token_agree": sum(r["agree"] for r in rs) / tok, "kl": sum(r["kl"] for r in rs) / tok,
            }
    return table


def markdown(table):
    lines = []
    for cond, systems in table.items():
        lines += [f"\n### {cond}\n", "| system | family | n | exact match | token agree | mean KL |",
                  "|---|---|---|---|---|---|"]
        for system, fams in systems.items():
            for fam in sorted(fams, key=lambda f: (f != "ALL", f)):
                m = fams[fam]
                lines.append(f"| {system} | {fam} | {m['n']} | {m['exact']:.3f} | {m['token_agree']:.3f} | {m['kl']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--teacher", default="data/teacher")
    ap.add_argument("--subset", default="data/flan_subset")
    ap.add_argument("--bridge", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--splits", nargs="+", default=["eval_seen", "eval_heldout"])
    ap.add_argument("--windows", nargs="+", default=["truncated", "full"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = run(args)
    table = summarise(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(results))
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    md = markdown(table)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
