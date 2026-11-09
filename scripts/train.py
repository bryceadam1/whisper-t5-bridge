"""Train the bridge by distillation from flan-t5-small on the text prompt (BLSP-style).

Stage 1 (default): only the bridge trains; Whisper and T5 are frozen.
Stage 2 (--lora-rank > 0, usually with --init-bridge from stage 1): also LoRA on T5's decoder
cross-attention (q, k, v, o).

Loss: KL(teacher || student) on the teacher's top-k next-token distributions along the teacher's own
output, plus --ce-weight times cross-entropy on the teacher's greedy tokens. Optional lexical terms:
  --repeat         one extra record per clip, "Write down exactly what was said:", whose target is the
                   spoken text itself (cross-entropy, weight --repeat-weight)
  --ctc-weight     CTC loss from a linear head on the bridge output to the spoken text's T5 tokens
                   (training only; the head is not part of the released bridge)
--injection picks where the bridge output enters T5 (see whisper_t5_bridge/bridge.py). Checkpoints are
selected on a 3% slice of the training clips, never on the eval clips.
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
from whisper_t5_bridge.data import ExampleSet, Examples, length_batches, to_device
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


def align_loss(t5, audio, audio_mask, batch):
    """Input-side alignment (after DiVA): bridge output i should match T5's encoder state for token i of the
    spoken text (encoded alone, with </s>). Relative MSE over positions both sequences have."""
    eos = t5.config.eos_token_id
    lens = batch["ctc_lens"] + 1
    ids = torch.zeros(len(lens), int(lens.max()), dtype=torch.long, device=audio.device)
    for j, n in enumerate(batch["ctc_lens"].tolist()):
        ids[j, :n] = batch["ctc_ids"][j, :n]
        ids[j, n] = eos
    text_mask = torch.arange(ids.shape[1], device=audio.device)[None, :] < lens[:, None]
    with torch.no_grad():
        target = encode_text(t5, ids, text_mask.long()).float()
    n = min(audio.shape[1], target.shape[1])
    both = (text_mask[:, :n] & audio_mask[:, :n]).float()[..., None]
    err = ((audio[:, :n].float() - target[:, :n]) ** 2 * both).sum()
    return err / ((target[:, :n] ** 2) * both).sum().clamp_min(1e-8)


def step_loss(t5, bridge, batch, w, autocast, ctc_head=None):
    """w: weights {"ce", "repeat", "ctc", "align"}. Returns total loss and a dict of its parts."""
    with autocast():
        enc, mask, audio, audio_mask = student_encoder_states(t5, bridge, batch, return_audio=True)
        logits = decoder_logits(t5, enc, mask, batch["labels"])
    valid = batch["labels"] >= 0
    kd = batch["kd"][:, None] & valid
    rep = ~batch["kd"][:, None] & valid
    tok_ce = F.cross_entropy(logits.float().transpose(1, 2), batch["labels"], ignore_index=-100, reduction="none")
    parts = {
        "kl": kl_topk(logits, batch["topk_ids"], batch["topk_logp"], kd),
        "ce": (tok_ce * kd).sum() / kd.sum().clamp_min(1),
        "repeat": (tok_ce * rep).sum() / rep.sum().clamp_min(1),
        "agree": ((logits.argmax(-1) == batch["labels"]) & kd).sum() / kd.sum().clamp_min(1),
    }
    loss = parts["kl"] + w["ce"] * parts["ce"] + w["repeat"] * parts["repeat"]
    if ctc_head is not None:
        logp = ctc_head(audio.float()).log_softmax(-1).transpose(0, 1)
        parts["ctc"] = F.ctc_loss(logp, batch["ctc_ids"], audio_mask.sum(1), batch["ctc_lens"],
                                  blank=ctc_head.out_features - 1, zero_infinity=True)
        loss = loss + w["ctc"] * parts["ctc"]
    if w.get("align"):
        parts["align"] = align_loss(t5, audio, audio_mask, batch)
        loss = loss + w["align"] * parts["align"]
    return loss, parts


@torch.no_grad()
def validate(t5, bridge, ex, batches, device, autocast):
    bridge.eval()
    tot = {"kl": 0.0, "ce": 0.0, "agree": 0.0}
    for idxs in batches:
        _, parts = step_loss(t5, bridge, to_device(ex.collate(idxs), device), {"ce": 0, "repeat": 0}, autocast)
        for k in tot:
            tot[k] += float(parts[k]) / len(batches)
    bridge.train()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--teacher", default="data/teacher")
    ap.add_argument("--out", default="runs/bridge")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=25600, help="cap on padded Whisper frames per batch")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ce-weight", type=float, default=0.2)
    ap.add_argument("--init-bridge", default=None)
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--time-masks", type=int, default=0, help="masked time spans per clip (0 = off)")
    ap.add_argument("--mask-frac", type=float, default=0.05)
    ap.add_argument("--state-noise", type=float, default=0.0)
    ap.add_argument("--injection", choices=["decoder", "encoder_input"], default="decoder")
    ap.add_argument("--n-conv", type=int, choices=[1, 2], default=1)
    ap.add_argument("--bridge-type", choices=["conv", "qformer"], default="conv",
                    help="qformer: queries through Whisper base.en's decoder layers, initialised from it")
    ap.add_argument("--queries-per-second", type=float, default=6.0)
    ap.add_argument("--pretrained-lr-scale", type=float, default=0.2)
    ap.add_argument("--align-weight", type=float, default=0.0, help="input-side alignment to T5 encoder states")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--repeat-weight", type=float, default=1.0)
    ap.add_argument("--ctc-weight", type=float, default=0.0)
    ap.add_argument("--subset", default="data/flan_subset")
    ap.add_argument("--extra", nargs="*", default=[], metavar="SPLIT:TEACHER_DIR:SUBSET_DIR",
                    help="more training sources, e.g. libri:data/teacher_libri:data/libri_subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok, t5 = load_t5(device)
    train = Examples(args.features, args.teacher, "train", tok, args.subset, part="train", repeat=args.repeat)
    if args.extra:
        sources = [train]
        for spec in args.extra:
            split, teacher_dir, subset_dir = spec.split(":")
            sources.append(Examples(args.features, teacher_dir, split, tok, subset_dir, part="train", repeat=args.repeat))
            print(f"extra source {split}: {len(sources[-1])} records over {len(sources[-1].meta)} clips", flush=True)
        train = ExampleSet(sources)
    val = Examples(args.features, args.teacher, "train", tok, args.subset, part="val")
    val_batches = length_batches(val, args.batch_size, shuffle=False, max_frames=args.max_frames)
    print(f"train {len(train)} records over {len(train.meta)} clips; val {len(val)} pairs over {len(val.meta)} "
          f"held-out training clips", flush=True)

    if args.init_bridge:
        bridge = load_bridge(args.init_bridge).to(device)
    else:
        if args.bridge_type == "qformer":
            from transformers import WhisperForConditionalGeneration
            from whisper_t5_bridge.qbridge import QBridge, QBridgeConfig

            whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
            bridge = QBridge.from_whisper(whisper, QBridgeConfig(queries_per_second=args.queries_per_second,
                                                                 dropout=args.dropout)).to(device)
            del whisper
        else:
            bridge = Bridge(BridgeConfig(n_layers=args.layers, dropout=args.dropout, n_conv=args.n_conv,
                                         injection=args.injection)).to(device)
        # Calibrate the final norm against what the bridge stands in for: T5 encoder outputs on real prompts,
        # or the input embeddings of the spoken inputs' tokens.
        sample = [train.records[i] for i in range(0, len(train), max(1, len(train) // 512))]
        key = "prompt" if args.injection == "decoder" else "input"
        enc = tok([r[key] for r in sample], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            target = (encode_text(t5, enc.input_ids, enc.attention_mask) if args.injection == "decoder"
                      else t5.shared(enc.input_ids))
            rms = init_output_scale(bridge, target, enc.attention_mask.bool())
        print(f"target RMS {rms:.3f} ({args.injection})", flush=True)
    if hasattr(bridge, "new_parameters"):
        # Whisper-initialised layers train gently; the new query, projection and norm at the full rate.
        new = {id(p) for p in bridge.new_parameters()}
        groups = [{"params": [p for p in bridge.parameters() if id(p) in new], "lr": args.lr},
                  {"params": [p for p in bridge.parameters() if id(p) not in new], "lr": args.lr * args.pretrained_lr_scale}]
    else:
        groups = [{"params": list(bridge.parameters()), "lr": args.lr}]
    ctc_head = None
    if args.ctc_weight:
        ctc_head = torch.nn.Linear(bridge.config.d_model, t5.config.vocab_size + 1).to(device)  # last = blank
        groups[0]["params"] += list(ctc_head.parameters())
    weights = {"ce": args.ce_weight, "repeat": args.repeat_weight, "ctc": args.ctc_weight, "align": args.align_weight}
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
        for idxs in length_batches(train, args.batch_size, shuffle=True, seed=args.seed + epoch,
                                   max_frames=args.max_frames):
            if step >= args.steps:
                break
            set_lr(step)
            batch = to_device(train.collate(idxs), device)
            if args.time_masks or args.state_noise:
                batch = augment(batch, args.time_masks, args.mask_frac, args.state_noise)
            loss, parts = step_loss(t5, bridge, batch, weights, autocast, ctc_head)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch, "loss": float(loss), **{k: float(v) for k, v in parts.items()},
                       "lr": opt.param_groups[0]["lr"], "min": (time.time() - t0) / 60}
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
