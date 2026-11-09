#!/usr/bin/env bash
# Local budget watchdog for a VM session: when the remote chain writes its done marker (OK or FAILED), or
# GPU spend reaches the cap, fetch results, runs, logs and the release folder, then stop the instance.
#   ops/watchdog.sh <done-marker> <cap-usd> <run> [<run> ...]
cd "$(dirname "$0")/.."
export SSH_PORT=${SSH_PORT:-443}
MARKER=$1; CAP=$2; shift 2; RUNS=("$@")
fetch() {
  for r in "${RUNS[@]}"; do
    mkdir -p results/$r runs/$r
    ops/oci.sh rsync ":whisper-t5-bridge/results/$r/" "results/$r/"
    ops/oci.sh rsync ":whisper-t5-bridge/runs/$r/" "runs/$r/"
  done
  mkdir -p logs release_v3
  ops/oci.sh rsync ":whisper-t5-bridge/logs/" logs/
  ops/oci.sh rsync ":whisper-t5-bridge/release_v3/" release_v3/
}
while true; do
  cost=$(ops/oci.sh hours | sed -n 's/.*compute \$\([0-9.]*\).*/\1/p')
  marker=$(ops/oci.sh ssh -o ConnectTimeout=20 "cat whisper-t5-bridge/$MARKER 2>/dev/null" </dev/null 2>/dev/null)
  echo "$(date -u +%H:%M:%SZ) cost \$$cost marker: ${marker:-none}"
  if [ -n "$marker" ]; then echo "chain finished: $marker"; fetch; ops/oci.sh stop; exit 0; fi
  if awk "BEGIN{exit !($cost >= $CAP)}"; then echo "budget cap reached"; fetch; ops/oci.sh stop; exit 0; fi
  sleep 120
done
