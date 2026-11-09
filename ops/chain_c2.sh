#!/usr/bin/env bash
# Experiment C2: a further ~80 h of LibriSpeech (11 of the 48 train-clean-360 shards on the Hugging Face
# mirror), continuing from C_base.
#   tmux new -d -s chainc2 "bash ops/chain_c2.sh"
set -eu
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=data/raw/librispeech360/LibriSpeech
.venv/bin/python -m scripts.librispeech_from_parquet --shards 0 1 2 3 4 5 6 7 8 9 10 --root $R > logs/c2.log 2>&1
.venv/bin/python -m scripts.librispeech_prepare --root $R --subsets train-clean-360 --split-name libri360 \
  --subset-out data/libri360_subset --seed 1 >> logs/c2.log 2>&1
.venv/bin/python -m scripts.teacher_cache --subset data/libri360_subset --split train --out data/teacher_libri360 \
  --batch-size 128 >> logs/c2.log 2>&1
.venv/bin/python -m scripts.whisper_features --split libri360 --skip-qc --audio $R --subset data/libri360_subset \
  --batch-size 32 2>&1 | grep --line-buffered -v -i warn >> logs/c2.log
.venv/bin/python -m scripts.train --out runs/C2 --init-bridge runs/C_base/bridge.safetensors --lr 3e-4 --warmup 200 \
  --dropout 0.2 --time-masks 2 --mask-frac 0.05 --state-noise 0.1 --steps 8000 --eval-every 1000 \
  --extra libri:data/teacher_libri:data/libri_subset libri360:data/teacher_libri360:data/libri360_subset \
  > logs/train_C2.log 2>&1
.venv/bin/python -m scripts.evaluate --bridge runs/C2/bridge.safetensors --out results/C2 > logs/eval_C2.log 2>&1
echo "C2_DONE $(date -u +%H:%M:%SZ)" > logs/c2.done
