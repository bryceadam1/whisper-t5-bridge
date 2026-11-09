"""Train the bridge by distillation from flan-t5-small on the text prompt (BLSP-style).

Stage 1 (default): only the bridge trains; Whisper and T5 are frozen.
Stage 2 (--lora-rank > 0, usually with --init-bridge from stage 1): also LoRA on T5's decoder
cross-attention (q, k, v, o).

Loss: KL(teacher || student) on the teacher's top-k next-token distributions along the teacher's own
output, plus --ce-weight times cross-entropy on the teacher's greedy tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from whisper_t5_bridge.bridge import Bridge, BridgeConfig, init_output_scale, load_bridge, param_count, save_bridge
from whisper_t5_bridge.data import Examples, length_batches, to_device
from whisper_t5_bridge.student import (add_cross_attn_lora, decoder_logits, encode_text, kl_topk, load_t5,
                                       lora_state, student_encoder_states)
from safetensors.torch import save_file


def augment(batch, n_masks, max_frac, noise):
    """SpecAugment-style regulariser on the Whisper states: zero n_masks random time spans, each up to
    max_frac of the clip, and add Gaussian noise of std `noise` (relative to the states' std)."""
    states, lens = batch["states"].clone(), batch["lens"]
    for j, n in enumerate(lens.tolist()):
        for _ in range(n_masks):
            width = int(torch.randint(0, max(1, int(max_frac * n)) + 1, ()).item())
            start = int(torch.randint(0, max(1, n - width) + 1, ()).item())
            states[j, start:start + width] = 0
    if noise:
        states = states + noise * states.std() * torch.randn_like(states)
    return {**batch, "states": states}


def step_loss(t5, bridge, batch, ce_weight, autocast):
    with autocast():
        enc, mask = student_encoder_states(t5, bridge, batch["instr_ids"], batch["instr_mask"],
                                           batch["states"], batch["lens"])
        logits = decoder_logits(t5, enc, mask, batch["labels"])
    valid = batch["labels"] >= 0
    kl = kl_topk(logits, batch["topk_ids"], batch["topk_logp"], valid)
    ce = F.cross_entropy(logits.float().transpose(1, 2), batch["labels"], ignore_index=-100)
    agree = ((logits.argmax(-1) == batch["labels"]) & valid).sum() / valid.sum()
    return kl + ce_weight * ce, kl, ce, agree


@torch.no_grad()
def validate(t5, bridge, ex, batches, device, autocast):
    bridge.eval()
    tot = {"kl": 0.0, "ce": 0.0, "agree": 0.0}
    for idxs in batches:
        b = to_device(ex.collate(idxs), device)
        _, kl, ce, agree = step_loss(t5, bridge, b, 0.0, autocast)
        for k, v in (("kl", kl), ("ce", ce), ("agree", agree)):
            tot[k] += float(v) / len(batches)
    bridge.train()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--teacher", default="data/teacher")
    ap.add_argument("--out", default="runs/bridge")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ce-weight", type=float, default=0.2)
    ap.add_argument("--init-bridge", default=None)
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-size", type=int, default=600)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--time-masks", type=int, default=0, help="masked time spans per clip (0 = off)")
    ap.add_argument("--mask-frac", type=float, default=0.05)
    ap.add_argument("--state-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok, t5 = load_t5(device)
    train = Examples(args.features, args.teacher, "train")
    val = Examples(args.features, args.teacher, "eval_seen")
    val_batches = length_batches(val, args.batch_size, shuffle=False)[: max(1, args.val_size // args.batch_size)]
    print(f"train {len(train)} pairs over {len(train.meta)} clips; val {sum(map(len, val_batches))} pairs", flush=True)

    if args.init_bridge:
        bridge = load_bridge(args.init_bridge).to(device)
    else:
        bridge = Bridge(BridgeConfig(n_layers=args.layers, dropout=args.dropout)).to(device)
        # Calibrate the final norm against T5 encoder outputs on real prompts.
        sample = [train.records[i]["prompt"] for i in range(0, len(train), max(1, len(train) // 512))]
        enc = tok(sample, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            rms = init_output_scale(bridge, encode_text(t5, enc.input_ids, enc.attention_mask), enc.attention_mask.bool())
        print(f"T5 encoder output RMS {rms:.3f}", flush=True)
    groups = [{"params": list(bridge.parameters()), "lr": args.lr}]
    adapters = {}
    if args.lora_rank:
        adapters = add_cross_attn_lora(t5, rank=args.lora_rank)
        groups.append({"params": [p for m in adapters.values() for p in (m.A, m.B)], "lr": args.lora_lr})
    n_lora = sum(p.numel() for m in adapters.values() for p in (m.A, m.B))
    print(f"bridge params {param_count(bridge):,}; LoRA params {n_lora:,}", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in groups]

    def set_lr(step):
        w = min(1.0, (step + 1) / args.warmup)
        c = 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * w * (0.05 + 0.95 * c)

    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "log.jsonl", "a")
    best, step, epoch, t0 = float("inf"), 0, 0, time.time()
    bridge.train()
    while step < args.steps:
        for idxs in length_batches(train, args.batch_size, shuffle=True, seed=args.seed + epoch):
            if step >= args.steps:
                break
            set_lr(step)
            batch = to_device(train.collate(idxs), device)
            if args.time_masks or args.state_noise:
                batch = augment(batch, args.time_masks, args.mask_frac, args.state_noise)
            loss, kl, ce, agree = step_loss(t5, bridge, batch, args.ce_weight, autocast)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch, "loss": float(loss), "kl": float(kl), "ce": float(ce),
                       "agree": float(agree), "lr": opt.param_groups[0]["lr"], "min": (time.time() - t0) / 60}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                v = validate(t5, bridge, val, val_batches, device, autocast)
                rec = {"step": step, "val_kl": v["kl"], "val_ce": v["ce"], "val_agree": v["agree"]}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print("VAL", rec, flush=True)
                if v["kl"] < best:
                    best = v["kl"]
                    save_bridge(bridge, str(out / "bridge.safetensors"),
                                {"step": step, "val": v, "train_args": vars(args)})
                    if adapters:
                        save_file(lora_state(adapters), str(out / "cross_attn_lora.safetensors"),
                                  metadata={"rank": str(args.lora_rank), "alpha": "16.0",
                                            "merge": "W' = W + (alpha / rank) * lora_B @ lora_A"})
        epoch += 1
    print(f"done: best val KL {best:.4f}, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
