"""Join teacher records with Whisper encoder states, and batch them by length."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

REPEAT_INSTRUCTION = "Write down exactly what was said:"
_STATES: dict[tuple[str, bool], np.ndarray] = {}  # one copy per feature file, shared by every source using it


def load_states(path: Path, in_memory: bool) -> np.ndarray:
    key = (str(path), in_memory)
    if key not in _STATES:
        arr = np.load(path, mmap_mode="r")
        # np.array copies a memmap into RAM (np.ascontiguousarray would return the memmap itself, leaving every
        # batch to fault pages in from disk at random).
        _STATES[key] = np.array(arr) if in_memory else arr
    return _STATES[key]
VAL_PERCENT = 3


def is_val(clip_id: str) -> bool:
    """Deterministic 3% of training clips held out for checkpoint selection."""
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16) % 100 < VAL_PERCENT


class Examples:
    """Teacher records (one per rendering) paired with one clip's encoder states.

    split: "train", "eval_seen" or "eval_heldout"; window: "truncated" or "full".
    part (train only): "train" excludes the validation clips, "val" keeps only them, None keeps all.
    repeat: also add one record per clip whose target is the spoken text itself (cross-entropy only).
    """

    def __init__(self, features_dir, teacher_dir, split, tokenizer, subset_dir="data/flan_subset",
                 window="truncated", keep_only=True, part=None, repeat=False, in_memory=True):
        features_dir, teacher_dir = Path(features_dir), Path(teacher_dir)
        self.meta = {}
        for line in open(features_dir / f"{split}_meta.jsonl"):
            m = json.loads(line)
            if (m.get("keep", True) or not keep_only) and (part is None or is_val(m["id"]) == (part == "val")):
                self.meta[m["id"]] = m
        src = "eval" if split.startswith("eval") else "train"
        spoken = {json.loads(l)["id"]: json.loads(l)["tts_text"] for l in open(Path(subset_dir) / f"{src}.jsonl")}
        # A subset may cover only some of the split's clips (e.g. generated tasks for 12k of 28k utterances).
        self.meta = {k: v for k, v in self.meta.items() if k in spoken}
        records = torch.load(teacher_dir / f"{src}.pt", weights_only=False)
        # Teacher outputs that never reached EOS are repetition loops; leave them out.
        self.records = [dict(r, kind="task") for r in records if r["id"] in self.meta and r["finished"]]
        if repeat:
            target = tokenizer([spoken[i] for i in self.meta]).input_ids
            instr = tokenizer(REPEAT_INSTRUCTION).input_ids
            self.records += [{"id": i, "kind": "repeat", "task": "repeat", "family": "repeat",
                              "prompt": f"{REPEAT_INSTRUCTION} {spoken[i]}", "input": spoken[i],
                              "instruction_ids": instr, "out_ids": torch.tensor(t, dtype=torch.int32)}
                             for i, t in zip(self.meta, target)]
        # Prompt text around the input (for encoder-input injection) and the spoken text's tokens (for CTC).
        befores, afters = [], []
        for r in self.records:
            before, sep, after = r["prompt"].partition(r["input"])
            if not sep:
                before, after = tokenizer.decode(r["instruction_ids"], skip_special_tokens=True), ""
            befores.append(before.strip())
            afters.append(after.strip())
        before_ids = tokenizer(befores, add_special_tokens=False).input_ids
        after_ids = tokenizer(afters, add_special_tokens=False).input_ids
        ctc = dict(zip(self.meta, tokenizer([spoken[i] for i in self.meta], add_special_tokens=False).input_ids))
        for r, b, a in zip(self.records, before_ids, after_ids):
            r["before_ids"], r["after_ids"], r["ctc_ids"] = b, a, ctc[r["id"]]
        self.states = load_states(features_dir / f"{split}_{window}.npy", in_memory)
        self.window = window

    def __len__(self):
        return len(self.records)

    def frames(self, idx):
        return self.meta[self.records[idx]["id"]][f"{self.window}_frames"]

    def get_states(self, clip_id):
        m = self.meta[clip_id]
        o, n = m[f"{self.window}_offset"], m[f"{self.window}_frames"]
        return self.states[o:o + n]

    def collate(self, idxs):
        recs = [self.records[i] for i in idxs]
        return make_batch(recs, [self.get_states(r["id"]) for r in recs])


class ExampleSet:
    """Several Examples sources (e.g. FLAN TTS clips and LibriSpeech) behind one index."""

    def __init__(self, sources: list[Examples]):
        self.sources = sources
        self.index = [(s, i) for s in sources for i in range(len(s))]
        self.records = [s.records[i] for s, i in self.index]
        self.meta = {k: v for s in sources for k, v in s.meta.items()}

    def __len__(self):
        return len(self.index)

    def frames(self, idx):
        s, i = self.index[idx]
        return s.frames(i)

    def collate(self, idxs):
        pairs = [self.index[i] for i in idxs]
        return make_batch([s.records[i] for s, i in pairs], [s.get_states(s.records[i]["id"]) for s, i in pairs])


def make_batch(recs, states, pad_id=0):
    """Pad one batch: Whisper states, instruction and prompt-text ids, targets, teacher top-k, CTC labels."""
    b, t = len(recs), max(len(s) for s in states)
    li = max(len(r["instruction_ids"]) for r in recs)
    lo = max(len(r["out_ids"]) for r in recs)
    lc = max(len(r["ctc_ids"]) for r in recs)
    k = next((r["topk_ids"].shape[1] for r in recs if r["kind"] == "task"), 1)
    batch = {
        "states": torch.zeros(b, t, 512), "lens": torch.tensor([len(s) for s in states]),
        "instr_ids": torch.full((b, li), pad_id), "instr_mask": torch.zeros(b, li, dtype=torch.long),
        "before": [r["before_ids"] for r in recs], "after": [r["after_ids"] for r in recs],
        "labels": torch.full((b, lo), -100), "topk_ids": torch.zeros(b, lo, k, dtype=torch.long),
        "topk_logp": torch.zeros(b, lo, k), "kd": torch.tensor([r["kind"] == "task" for r in recs]),
        "ctc_ids": torch.zeros(b, lc, dtype=torch.long), "ctc_lens": torch.tensor([len(r["ctc_ids"]) for r in recs]),
        "records": recs,
    }
    for j, (r, s) in enumerate(zip(recs, states)):
        batch["states"][j, : len(s)] = torch.from_numpy(np.asarray(s, dtype=np.float32))
        ids = r["instruction_ids"]
        batch["instr_ids"][j, : len(ids)] = torch.tensor(ids)
        batch["instr_mask"][j, : len(ids)] = 1
        n = len(r["out_ids"])
        batch["labels"][j, :n] = r["out_ids"].long()
        if r["kind"] == "task":
            batch["topk_ids"][j, :n] = r["topk_ids"].long()
            batch["topk_logp"][j, :n] = r["topk_logp"].float()
        batch["ctc_ids"][j, : len(r["ctc_ids"])] = torch.tensor(r["ctc_ids"])
    return batch


def length_batches(ex: Examples, batch_size: int, shuffle: bool, seed: int = 0, max_frames: int | None = None):
    """Batches of similar audio length (sorted within shuffled pools of 50 batches), of at most batch_size
    examples and, if max_frames is set, at most max_frames padded Whisper frames."""
    rng = random.Random(seed)
    idx = list(range(len(ex)))
    if shuffle:
        rng.shuffle(idx)
    pool = batch_size * 50
    batches = []
    for p in range(0, len(idx), pool):
        current = []
        for i in sorted(idx[p:p + pool], key=ex.frames):
            # Sorted ascending, so the newest example sets the padded length.
            if current and (len(current) == batch_size or
                            (max_frames and ex.frames(i) * (len(current) + 1) > max_frames)):
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    return batches


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
