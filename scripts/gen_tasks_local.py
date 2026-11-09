"""FLAN-style, content-dependent tasks for transcripts, written by a small open model (no API usage).

The model (Qwen2.5-1.5B-Instruct, Apache-2.0, pinned) only writes the content for each transcript, in a
fixed line format: a question and its answer span, three wrong answers, and a true and a false
statement. The prompts themselves are assembled here from FLAN-style wordings, so every template has a
valid format. Items that fail checks (answer not in the transcript, answer leaked into the question,
missing fields) are dropped for that family.

Input: JSONL with {"id", "transcript"}. Output: JSONL with {"id", "family", "template", "gold"}, the
transcript replaced by {x}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

INSTRUCTIONS = """Read the text and write exactly these six lines about it:
Q: a short question whose answer is a word or short phrase copied from the text (the question must not contain the answer)
A: the answer, copied exactly from the text
WRONG: three plausible wrong answers of the same kind, separated by |
TRUE: a short statement that the text says is true
FALSE: the same statement with one detail (a name, number, object or action) changed so that it is false
"""
EXAMPLE_TEXT = "the old fisherman pulled his boat onto the sand and sat down beside his two sons"
EXAMPLE_OUT = """Q: Where did the fisherman pull his boat?
A: onto the sand
WRONG: into the river | under the bridge | onto the rocks
TRUE: the fisherman sat down beside his two sons
FALSE: the fisherman sat down beside his three daughters"""

QA = ["Read the text and answer the question.\n\n{x}\n\nQuestion: {q}\nAnswer:",
      "{x}\n\nAnswer this question based on the passage: {q}",
      "Context: {x}\n\nQ: {q}\nA:",
      "{x}\n\nBased on the text above, {q_lower}"]
MC = ["{x}\n\nAccording to the text, {q_lower}\nOPTIONS:\n{opts}",
      "Read the passage and choose the correct answer.\n\n{x}\n\nQuestion: {q}\nOPTIONS:\n{opts}",
      "{x}\n\n{q}\nChoose from:\n{opts}"]
VER = ["{x}\n\nBased on the text above, is it true that \"{s}\"?\nOPTIONS:\n- yes\n- no",
       "Read the text and decide whether the statement is true.\n\nText: {x}\nStatement: {s}\nOPTIONS:\n- yes\n- no",
       "{x}\n\nDoes it follow from the text that {s}?\nOPTIONS:\n- yes\n- no"]


def parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(Q|A|WRONG|TRUE|FALSE)\s*:\s*(.+)", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2).strip()
    return out


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def build(item: dict, fields: dict, rng: random.Random) -> list[dict]:
    t, tasks = norm(item["transcript"]), []
    q, a = fields.get("Q", ""), fields.get("A", "").strip(" .")
    if q and a and norm(a) and norm(a) in t and norm(a) not in norm(q):
        q = q if q.endswith("?") else q + "?"
        ql = q[0].lower() + q[1:]
        tasks.append({"family": "extractive_qa", "template": rng.choice(QA).format(x="{x}", q=q, q_lower=ql), "gold": a})
        wrong = [w.strip(" .") for w in fields.get("WRONG", "").split("|") if w.strip(" .")]
        wrong = [w for w in wrong if norm(w) and norm(w) != norm(a)][:3]
        if len(wrong) == 3:
            opts = wrong + [a]
            rng.shuffle(opts)
            tasks.append({"family": "multiple_choice", "gold": a, "template": rng.choice(MC).format(
                x="{x}", q=q, q_lower=ql, opts="\n".join(f"- {o}" for o in opts))})
    true, false = fields.get("TRUE", "").strip(" ."), fields.get("FALSE", "").strip(" .")
    if true and false and norm(true) != norm(false):
        s, gold = (true, "yes") if rng.random() < 0.5 else (false, "no")
        tasks.append({"family": "verification", "gold": gold, "template": rng.choice(VER).format(x="{x}", s=s)})
    return [dict(t, id=item["id"]) for t in tasks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16 if device == "cuda"
                                                 else torch.float16).to(device).eval()
    items = [json.loads(l) for f in args.inputs for l in open(f)]
    rng, t0, n_tasks = random.Random(args.seed), time.time(), 0
    with open(args.out, "w") as fh:
        for b in range(0, len(items), args.batch_size):
            batch = items[b:b + args.batch_size]
            prompts = [tok.apply_chat_template([
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Text: {EXAMPLE_TEXT}"}, {"role": "assistant", "content": EXAMPLE_OUT},
                {"role": "user", "content": f"Text: {it['transcript']}"}], tokenize=False, add_generation_prompt=True)
                for it in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=160, do_sample=False)
            texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for it, text in zip(batch, texts):
                for task in build(it, parse(text), rng):
                    fh.write(json.dumps(task) + "\n")
                    n_tasks += 1
            print(f"{b + len(batch)}/{len(items)} transcripts, {n_tasks} tasks, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
