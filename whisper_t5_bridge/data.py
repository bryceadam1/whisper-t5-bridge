"""Join teacher records with Whisper encoder states, and batch them by length."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


class Examples:
    """Teacher records (one per rendering) paired with one clip's encoder states.

    split: "train", "eval_seen" or "eval_heldout"; window: "truncated" or "full"."""

    def __init__(self, features_dir, teacher_dir, split, window="truncated", keep_only=True,
                 renderings=None, in_memory=True):
        features_dir, teacher_dir = Path(features_dir), Path(teacher_dir)
        self.meta = {}
        for line in open(features_dir / f"{split}_meta.jsonl"):
            m = json.loads(line)
            if m.get("keep", True) or not keep_only:
                self.meta[m["id"]] = m
        records = torch.load(teacher_dir / ("train.pt" if split == "train" else "eval.pt"), weights_only=False)
        # Teacher outputs that never reached EOS are repetition loops; leave them out.
        self.records = [r for r in records if r["id"] in self.meta and r["finished"]
                        and (renderings is None or r["rendering"] < renderings)]
        arr = np.load(features_dir / f"{split}_{window}.npy", mmap_mode="r")
        self.states = np.ascontiguousarray(arr) if in_memory else arr
        self.window = window

    def __len__(self):
        return len(self.records)

    def frames(self, idx):
        return self.meta[self.records[idx]["id"]][f"{self.window}_frames"]

    def get_states(self, clip_id):
        m = self.meta[clip_id]
        o, n = m[f"{self.window}_offset"], m[f"{self.window}_frames"]
        return self.states[o:o + n]

    def collate(self, idxs, pad_id=0):
        recs = [self.records[i] for i in idxs]
        states = [self.get_states(r["id"]) for r in recs]
        b, t = len(recs), max(len(s) for s in states)
        li = max(len(r["instruction_ids"]) for r in recs)
        lo = max(len(r["out_ids"]) for r in recs)
        k = recs[0]["topk_ids"].shape[1]
        batch = {
            "states": torch.zeros(b, t, 512), "lens": torch.tensor([len(s) for s in states]),
            "instr_ids": torch.full((b, li), pad_id), "instr_mask": torch.zeros(b, li, dtype=torch.long),
            "labels": torch.full((b, lo), -100), "topk_ids": torch.zeros(b, lo, k, dtype=torch.long),
            "topk_logp": torch.zeros(b, lo, k), "records": recs,
        }
        for j, (r, s) in enumerate(zip(recs, states)):
            batch["states"][j, : len(s)] = torch.from_numpy(np.asarray(s, dtype=np.float32))
            ids = r["instruction_ids"]
            batch["instr_ids"][j, : len(ids)] = torch.tensor(ids)
            batch["instr_mask"][j, : len(ids)] = 1
            n = len(r["out_ids"])
            batch["labels"][j, :n] = r["out_ids"].long()
            batch["topk_ids"][j, :n] = r["topk_ids"].long()
            batch["topk_logp"][j, :n] = r["topk_logp"].float()
        return batch


def length_batches(ex: Examples, batch_size: int, shuffle: bool, seed: int = 0):
    """Batches of similar audio length (sorted within shuffled pools of 50 batches)."""
    rng = random.Random(seed)
    idx = list(range(len(ex)))
    if shuffle:
        rng.shuffle(idx)
    pool = batch_size * 50
    batches = []
    for p in range(0, len(idx), pool):
        part = sorted(idx[p:p + pool], key=ex.frames)
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
