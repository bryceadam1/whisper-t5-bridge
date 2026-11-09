#!/usr/bin/env bash
# Experiment C: add LibriSpeech (real speech) to the FLAN TTS data; baseline recipe and encoder-input
# injection trained concurrently, each evaluated when done.
#   tmux new -d -s chainc "bash ops/chain_c.sh"
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--dropout 0.2 --time-masks 2 --mask-frac 0.05 --state-noise 0.1 --steps 12000 --eval-every 1000
        --extra libri:data/teacher_libri:data/libri_subset"

run() {  # name, extra train args
  local name=$1; shift
  rm -rf "runs/$name"
  .venv/bin/python -m scripts.train --out "runs/$name" $COMMON "$@" > "logs/train_$name.log" 2>&1
  echo "TRAIN_DONE $(date -u +%H:%M:%SZ)" >> "logs/train_$name.log"
  .venv/bin/python -m scripts.evaluate --bridge "runs/$name/bridge.safetensors" --out "results/$name" \
    > "logs/eval_$name.log" 2>&1
  echo "EVAL_DONE $(date -u +%H:%M:%SZ)" >> "logs/eval_$name.log"
}

run C_base &
sleep 120  # stagger start-up (both load ~30 GB of states into RAM)
run C_encoder_input --injection encoder_input &
wait
echo "CHAIN_DONE $(date -u +%H:%M:%SZ)" > logs/chain_c.done
