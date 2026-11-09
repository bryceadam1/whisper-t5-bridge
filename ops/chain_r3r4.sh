#!/usr/bin/env bash
# Round 3 session: FLAN-style tasks on real speech, then R3 (conv bridge from C2) and R4 (Whisper-decoder
# Q-Former at 25 queries/s) trained concurrently, evaluated, and the best bridge exported.
#   tmux new -d -s chain "bash ops/chain_r3r4.sh"
# Writes logs/chain_r3r4.done containing OK or FAILED: <step>.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
DONE=logs/chain_r3r4.done
rm -f $DONE
fail() { echo "FAILED: $1" > $DONE; exit 1; }
mkdir -p logs data/libri360_flan

# Page cache warm-up (sequential reads), so training does not start I/O-bound.
(for f in train libri libri360; do dd if=data/features/${f}_truncated.npy of=/dev/null bs=16M status=none; done) &

# 1. FLAN sentence-level templates on every LibriSpeech transcript.
$PY -m scripts.flan_templates_on_speech --subset data/libri_subset --out data/libri_tmpl_subset > logs/prep_tmpl.log 2>&1 || fail tmpl100
$PY -m scripts.flan_templates_on_speech --subset data/libri360_subset --out data/libri360_tmpl_subset --seed 1 >> logs/prep_tmpl.log 2>&1 || fail tmpl360

# 2. Qwen tasks for 12k train-clean-360 transcripts (GPU), in the background.
$PY - <<'EOF'
import json, random
rows = [json.loads(l) for l in open("data/libri360_subset/train.jsonl")]
rows = [{"id": r["id"], "transcript": r["input"]} for r in rows if 6 <= len(r["input"].split()) <= 60]
random.Random(1).shuffle(rows)
with open("data/libri360_flan/transcripts_12k.jsonl", "w") as fh:
    for r in rows[:12000]:
        fh.write(json.dumps(r) + "\n")
EOF
( $PY -m scripts.gen_tasks_local --inputs data/libri360_flan/transcripts_12k.jsonl --out data/libri360_flan/tasks.jsonl --batch-size 64 \
  && $PY -m scripts.build_task_subset --tasks data/libri360_flan/tasks.jsonl --transcripts data/libri360_flan/transcripts_12k.jsonl --out data/libri360_flan_subset \
  && $PY -m scripts.teacher_cache --subset data/libri360_flan_subset --split train --out data/teacher_libri360_flan --batch-size 128 \
  && echo QWEN_OK ) > logs/prep_qwen.log 2>&1 &
QWEN=$!

# 3. Teacher outputs for the template prompts (GPU, alongside the Qwen job).
$PY -m scripts.teacher_cache --subset data/libri_tmpl_subset --split train --out data/teacher_libri_tmpl --batch-size 128 >> logs/prep_tmpl.log 2>&1 || fail teacher_tmpl100
$PY -m scripts.teacher_cache --subset data/libri360_tmpl_subset --split train --out data/teacher_libri360_tmpl --batch-size 128 >> logs/prep_tmpl.log 2>&1 || fail teacher_tmpl360
wait $QWEN
grep -q QWEN_OK logs/prep_qwen.log || fail qwen
wait

EXTRA="libri:data/teacher_libri:data/libri_subset libri360:data/teacher_libri360:data/libri360_subset
       libri:data/teacher_libri_flan:data/libri_flan_subset libri360:data/teacher_libri360_flan:data/libri360_flan_subset
       libri:data/teacher_libri_tmpl:data/libri_tmpl_subset libri360:data/teacher_libri360_tmpl:data/libri360_tmpl_subset"
AUG="--dropout 0.2 --time-masks 2 --mask-frac 0.05 --state-noise 0.1 --steps 10000 --eval-every 1000"

run() {  # name, train args
  local name=$1; shift
  $PY -m scripts.train --out runs/$name $AUG "$@" --extra $EXTRA > logs/train_$name.log 2>&1
  [ -f runs/$name/bridge.safetensors ] || { echo "TRAIN_FAILED" >> logs/train_$name.log; return 1; }
  $PY -m scripts.evaluate --bridge runs/$name/bridge.safetensors --out results/$name > logs/eval_$name.log 2>&1
  [ -f results/$name/summary.json ] || { echo "EVAL_FAILED" >> logs/eval_$name.log; return 1; }
}

# 4. R3 and R4 concurrently (staggered so they do not load features at the same moment).
run R3_flan_speech --init-bridge runs/C2/bridge.safetensors --lr 3e-4 --warmup 200 &
R3=$!
sleep 240
run R4_qformer25 --bridge-type qformer --queries-per-second 25 --pretrained-lr-scale 1.0 --align-weight 0 &
R4=$!
wait $R3; s3=$?
wait $R4; s4=$?

# 5. Pick the best bridge on held-out voices, truncated window (exact match, then KL) and export it.
best=$($PY - <<'EOF'
import json, os
best = None
for r in ("C2", "R3_flan_speech", "R4_qformer25"):
    p = f"results/{r}/summary.json"
    if os.path.exists(p):
        a = json.load(open(p))["eval_heldout/truncated"]["bridge"]["ALL"]
        key = (round(a["exact"], 3), -a["kl"])
        if best is None or key > best[0]:
            best = (key, r)
print(best[1])
EOF
)
$PY -m scripts.export --bridge runs/$best/bridge.safetensors --out release_v3 > logs/export.log 2>&1 && echo "$best" > release_v3/SOURCE
echo "OK best=$best r3=$s3 r4=$s4" > $DONE
