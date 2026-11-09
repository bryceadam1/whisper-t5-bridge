#!/usr/bin/env bash
# Manage the single GPU instance for this project and keep a running total of GPU hours.
#
#   ops/oci.sh launch | start | stop | status | ip | ssh [cmd...] | rsync SRC DST | hours | allow-me | enable-443
#
# Resource ids live in ~/.config/whisper-t5-bridge/oci.env (outside the repo); OCI credentials in
# ~/.oci. Every start/stop is appended to ops/usage_log.csv. Billing for VM.GPU.A10 shapes pauses while
# the instance is STOPPED; the boot volume keeps billing until the instance is terminated.
set -euo pipefail
export SUPPRESS_LABEL_WARNING=True PYTHONWARNINGS=ignore

ENV=~/.config/whisper-t5-bridge/oci.env
source "$ENV"
HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$HERE/usage_log.csv"
RATE=2.00  # USD per hour, VM.GPU.A10.1 list price
SHAPE=VM.GPU.A10.1
AD=LIGD:PHX-AD-1
IMAGE=ocid1.image.oc1.phx.aaaaaaaasxj5lm6nn7h6udnyzjw4slshdoz56473tdy65ly6r4umy5aa3q6q  # Oracle-Linux-9.8-Gen2-GPU-2026.10.30-0
BOOT_GB=150
SSH_PORT=${SSH_PORT:-22}  # 443 also works once `enable-443` has run (for networks that block outbound 22)
# One multiplexed connection per host: networks that rate-limit SSH see a single TCP connection.
SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ControlMaster=auto
          -o ControlPath=$HOME/.ssh/cm-whisper-t5-bridge-%p -o ControlPersist=30m"

[ -f "$LOG" ] || echo "utc_time,event,instance_state" > "$LOG"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
state() { oci compute instance get --instance-id "$INSTANCE" --query 'data."lifecycle-state"' --raw-output; }
ip() {
  local vnic
  vnic=$(oci compute vnic-attachment list -c "$COMPARTMENT" --instance-id "$INSTANCE" --query 'data[0]."vnic-id"' --raw-output)
  oci network vnic get --vnic-id "$vnic" --query 'data."public-ip"' --raw-output
}

hours() {
  python3 - "$LOG" "$RATE" <<'EOF'
import csv, sys
from datetime import datetime, timezone
rows = list(csv.DictReader(open(sys.argv[1])))
total, start = 0.0, None
for r in rows:
    t = datetime.strptime(r["utc_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if r["event"] in ("launch", "start") and start is None:
        start = t
    elif r["event"] == "stop" and start is not None:
        total += (t - start).total_seconds(); start = None
running = ""
if start is not None:
    total += (datetime.now(timezone.utc) - start).total_seconds(); running = " (instance running now)"
h = total / 3600
print(f"GPU time {h:.2f} h, compute ${h * float(sys.argv[2]):.2f}{running}")
EOF
}

case "${1:-status}" in
  launch)
    [ -z "${INSTANCE:-}" ] || { echo "already launched: $INSTANCE"; exit 1; }
    INSTANCE=$(oci compute instance launch -c "$COMPARTMENT" --availability-domain "$AD" --shape "$SHAPE" \
      --image-id "$IMAGE" --subnet-id "$SUBNET" --assign-public-ip true --display-name bridge-gpu \
      --boot-volume-size-in-gbs "$BOOT_GB" --ssh-authorized-keys-file ~/.ssh/id_ed25519.pub \
      --wait-for-state RUNNING --query data.id --raw-output)
    echo "INSTANCE=$INSTANCE" >> "$ENV"
    echo "$(now),launch,RUNNING" >> "$LOG"
    echo "launched $INSTANCE at $(ip)"; hours ;;
  start)
    oci compute instance action --instance-id "$INSTANCE" --action START --wait-for-state RUNNING >/dev/null
    echo "$(now),start,RUNNING" >> "$LOG"; echo "running at $(ip)"; hours ;;
  stop)
    oci compute instance action --instance-id "$INSTANCE" --action SOFTSTOP --wait-for-state STOPPED >/dev/null
    echo "$(now),stop,STOPPED" >> "$LOG"; echo "stopped"; hours ;;
  status) echo "$(state)"; hours ;;
  ip) ip ;;
  ssh) shift; exec ssh $SSH_OPTS "opc@$(ip)" "$@" ;;
  rsync)  # ops/oci.sh rsync <src> <dst>, remote paths written as :path
    shift; host="opc@$(ip)"; exec rsync -az -e "ssh $SSH_OPTS" "${1/#:/$host:}" "${2/#:/$host:}" ;;
  enable-443)  # make sshd also listen on 443 (SELinux label, firewalld rule), then allow it in the security list
    ssh $SSH_OPTS "opc@$(ip)" 'sudo grep -q "^Port 443" /etc/ssh/sshd_config || printf "Port 22\nPort 443\n" | sudo tee -a /etc/ssh/sshd_config >/dev/null;
      sudo semanage port -m -t ssh_port_t -p tcp 443 2>/dev/null || sudo semanage port -a -t ssh_port_t -p tcp 443;
      sudo firewall-cmd -q --permanent --add-port=443/tcp && sudo firewall-cmd -q --reload; sudo systemctl restart sshd && sudo ss -ltnp | grep sshd'
    echo "sshd listening on 22 and 443" ;;
  hours) hours ;;
  allow-me)  # point the SSH-only firewall rule at this machine's current public IP
    me=$(curl -s https://checkip.amazonaws.com)
    oci network security-list update --security-list-id "$SECLIST" --force --wait-for-state AVAILABLE \
      --ingress-security-rules "[{\"source\":\"$me/32\",\"protocol\":\"6\",\"isStateless\":false,\"tcpOptions\":{\"destinationPortRange\":{\"min\":22,\"max\":22}}},
                                 {\"source\":\"$me/32\",\"protocol\":\"6\",\"isStateless\":false,\"tcpOptions\":{\"destinationPortRange\":{\"min\":443,\"max\":443}}}]" >/dev/null
    echo "SSH (22, 443) allowed from $me" ;;
  *) echo "usage: $0 launch|start|stop|status|ip|ssh|rsync|hours|allow-me|enable-443"; exit 1 ;;
esac
