#!/usr/bin/env bash
# One-time OCI setup, run by the tenancy owner in OCI Cloud Shell. Safe to re-run: it reuses
# whatever already exists and creates only what is missing.
# Creates a sandbox compartment and a restricted "bridge-agent" API user whose only rights are
# compute, networking and volumes inside that compartment. Uploads the agent's PUBLIC key only.
set -uo pipefail
echo "bootstrap v3"

NAME=bridge-agent
COMP_NAME=whisper-t5-bridge
FINGERPRINT=cb:cb:e6:09:d9:5f:6e:54:2c:87:6f:ee:22:f5:b5:81

# Prints the first id matching a query, or nothing.
first_id() { "$@" --query 'data[0].id' --raw-output 2>/dev/null | grep '^ocid1' || true; }
die() { echo "FAILED: $1"; exit 1; }

COMP=$(first_id oci iam compartment list --compartment-id "$OCI_TENANCY" --name "$COMP_NAME" --lifecycle-state ACTIVE)
if [ -n "$COMP" ]; then echo "compartment: exists"; else
  COMP=$(oci iam compartment create --compartment-id "$OCI_TENANCY" --name "$COMP_NAME" \
    --description "Sandbox for whisper-t5-bridge GPU work" --wait-for-state ACTIVE \
    --query data.id --raw-output 2>/dev/null) || die "compartment create"
  echo "compartment: created"
fi

GRP=$(first_id oci iam group list --compartment-id "$OCI_TENANCY" --name "$NAME")
if [ -n "$GRP" ]; then echo "group: exists"; else
  GRP=$(oci iam group create --name "$NAME" --description "Claude agent for whisper-t5-bridge" \
    --query data.id --raw-output 2>/dev/null) || die "group create"
  echo "group: created"
fi

USR=$(first_id oci iam user list --compartment-id "$OCI_TENANCY" --name "$NAME")
if [ -n "$USR" ]; then echo "user: exists"; else
  ERR=$(mktemp)
  USR=$(oci iam user create --name "$NAME" --description "Claude agent for whisper-t5-bridge" \
    --query data.id --raw-output 2>"$ERR")
  if [ -z "$USR" ]; then
    # Identity-domain tenancies require an email on new users: use a +alias of the owner's own address.
    OWNER=$(oci iam user get --user-id "${OCI_CS_USER_OCID:-}" --query data.email --raw-output 2>/dev/null)
    if [[ "$OWNER" == *@* ]]; then
      USR=$(oci iam user create --name "$NAME" --description "Claude agent for whisper-t5-bridge" \
        --email "${OWNER%@*}+$NAME@${OWNER#*@}" --query data.id --raw-output 2>"$ERR")
    fi
  fi
  [ -n "$USR" ] || { grep -E '"(code|message)"' "$ERR" || tail -5 "$ERR"; echo "owner email lookup: ${OWNER:-none}"; die "user create"; }
  echo "user: created"
fi

if oci iam group list-users --group-id "$GRP" --all --query 'data[].id' 2>/dev/null | grep -q "$USR"; then
  echo "membership: exists"
else
  oci iam group add-user --group-id "$GRP" --user-id "$USR" >/dev/null 2>&1 || die "add user to group"
  echo "membership: added"
fi

if oci iam user api-key list --user-id "$USR" --query 'data[].fingerprint' 2>/dev/null | grep -q "$FINGERPRINT"; then
  echo "api key: exists"
else
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
  oci iam user api-key upload --user-id "$USR" --key-file "$KEY" >/dev/null 2>&1 || die "api key upload"
  echo "api key: uploaded"
fi

POL=$(first_id oci iam policy list --compartment-id "$OCI_TENANCY" --name "$NAME-policy")
if [ -n "$POL" ]; then echo "policy: exists"; else
  oci iam policy create --compartment-id "$OCI_TENANCY" --name "$NAME-policy" \
    --description "bridge-agent may only manage compute, network and volumes in whisper-t5-bridge" \
    --statements "[\"Allow group $NAME to manage instance-family in compartment $COMP_NAME\",
                   \"Allow group $NAME to manage virtual-network-family in compartment $COMP_NAME\",
                   \"Allow group $NAME to manage volume-family in compartment $COMP_NAME\"]" \
    >/dev/null 2>&1 || die "policy create"
  echo "policy: created"
fi

# Ids printed in short chunks, each with a 6-character checksum, so they can be read off a phone screen.
show() { echo; echo "$1 (check $(printf %s "$2" | md5sum | cut -c1-6)):"; printf %s "$2" | fold -w 24; echo; }
echo; echo "===== send this to Claude ====="
echo "region=$OCI_REGION"
show tenancy "$OCI_TENANCY"
show user "$USR"
show compartment "$COMP"
