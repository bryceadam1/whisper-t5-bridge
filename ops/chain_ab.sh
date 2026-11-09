#!/usr/bin/env bash
# Experiments A (lexical objectives) and B (encoder-input injection), run one after another on the VM;
# each evaluation runs in the background while the next model trains.
#   tmux new -d -s chain "bash ops/chain_ab.sh"
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AUG="--dropout 0.2 --time-masks 2 --mask-frac 0.05 --state-noise 0.1 --steps 6000 --eval-every 500"
mkdir -p logs

run() {  # name, extra train args
  local name=$1; shift
  rm -rf "runs/$name"
  .venv/bin/python -m scripts.train --out "runs/$name" $AUG "$@" > "logs/train_$name.log" 2>&1
  echo "TRAIN_DONE $(date -u +%H:%M:%SZ)" >> "logs/train_$name.log"
  ( .venv/bin/python -m scripts.evaluate --bridge "runs/$name/bridge.safetensors" --out "results/$name" \
      > "logs/eval_$name.log" 2>&1; echo "EVAL_DONE $(date -u +%H:%M:%SZ)" >> "logs/eval_$name.log" ) &
}

run B_encoder_input --injection encoder_input
run AB --injection encoder_input --repeat --ctc-weight 0.2
run A_lexical --repeat --ctc-weight 0.2
wait
echo "CHAIN_DONE $(date -u +%H:%M:%SZ)" > logs/chain_ab.done
