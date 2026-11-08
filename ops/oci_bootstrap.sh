#!/usr/bin/env bash
# One-time OCI setup, run by the tenancy owner in OCI Cloud Shell.
# Creates a sandbox compartment and a restricted "bridge-agent" API user whose only rights are
# compute, networking and volumes inside that compartment. Uploads the agent's PUBLIC key only.
set -euo pipefail

COMP=$(oci iam compartment create --compartment-id "$OCI_TENANCY" --name whisper-t5-bridge \
  --description "Sandbox for whisper-t5-bridge GPU work" --wait-for-state ACTIVE --query data.id --raw-output)
GRP=$(oci iam group create --name bridge-agent --description "Claude agent for whisper-t5-bridge" --query data.id --raw-output)
USR=$(oci iam user create --name bridge-agent --description "Claude agent for whisper-t5-bridge" --query data.id --raw-output)
oci iam group add-user --group-id "$GRP" --user-id "$USR" >/dev/null

KEY=$(mktemp)
cat > "$KEY" <<'EOF'
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyjqIUoNZRBqD3J39xpuQ
wGhcocJtkq3Bk5P2hepFHMahCyM2FxTF+eQ9n5Pp+gl+fw1wIAmQWTt/ZV/83OvG
nGdcZ/pJW8POZ7CoeouFq9vN2bFmzkjk9jbHOPtIpuxhHQTUcn5uB/a0ePiAC1VL
eNp2vW1wYC9kaGpq0/utsARejLD3+2MoKhodkR/yQPl8qxtQdPETrtnAii7GDikb
urqcul/7X6f9fisHBG9pDes7zhFJIXJ1zaNN8G+YkoUmBFQtoluI4sEPm4aHu/w5
rCiN4pICOCnImcBqaT7DfvkEd9Zm6oa/ubQLKLzzha2/D6daDK1odqCDqZqz2jx4
dQIDAQAB
-----END PUBLIC KEY-----
EOF
oci iam user api-key upload --user-id "$USR" --key-file "$KEY" >/dev/null

oci iam policy create --compartment-id "$OCI_TENANCY" --name bridge-agent-policy \
  --description "bridge-agent may only manage compute, network and volumes in whisper-t5-bridge" \
  --statements '["Allow group bridge-agent to manage instance-family in compartment whisper-t5-bridge",
                 "Allow group bridge-agent to manage virtual-network-family in compartment whisper-t5-bridge",
                 "Allow group bridge-agent to manage volume-family in compartment whisper-t5-bridge"]' >/dev/null

echo "tenancy=$OCI_TENANCY"
echo "region=$OCI_REGION"
echo "user=$USR"
echo "compartment=$COMP"
