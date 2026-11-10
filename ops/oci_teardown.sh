#!/usr/bin/env bash
# Remove what ops/oci_bootstrap.sh created, run by the tenancy owner in OCI Cloud Shell: the bridge-agent
# policy, user (with its API keys) and group, and the whisper-t5-bridge compartment. Safe to re-run.
# The compartment must hold no live resources (the instance, volumes and network were deleted first).
set -uo pipefail
echo "teardown v1"
first_id() { "$@" --query 'data[0].id' --raw-output 2>/dev/null | grep '^ocid1' || true; }

POL=$(first_id oci iam policy list --compartment-id "$OCI_TENANCY" --name bridge-agent-policy)
[ -n "$POL" ] && oci iam policy delete --policy-id "$POL" --force >/dev/null 2>&1 && echo "policy: deleted" || echo "policy: none"

USR=$(first_id oci iam user list --compartment-id "$OCI_TENANCY" --name bridge-agent)
GRP=$(first_id oci iam group list --compartment-id "$OCI_TENANCY" --name bridge-agent)
if [ -n "$USR" ]; then
  for fp in $(oci iam user api-key list --user-id "$USR" --query 'data[].fingerprint' --raw-output 2>/dev/null | tr -d '[]",'); do
    oci iam user api-key delete --user-id "$USR" --fingerprint "$fp" --force >/dev/null 2>&1 && echo "api key $fp: deleted"
  done
  if [ -n "$GRP" ]; then
    oci iam group remove-user --group-id "$GRP" --user-id "$USR" --force >/dev/null 2>&1 \
      && echo "membership: removed" || echo "membership: none"
  fi
  oci iam user delete --user-id "$USR" --force >/dev/null 2>&1 && echo "user: deleted" || echo "user: FAILED to delete"
else
  echo "user: none"
fi
[ -n "$GRP" ] && { oci iam group delete --group-id "$GRP" --force >/dev/null 2>&1 && echo "group: deleted" || echo "group: FAILED to delete"; } || echo "group: none"

COMP=$(first_id oci iam compartment list --compartment-id "$OCI_TENANCY" --name whisper-t5-bridge --lifecycle-state ACTIVE)
if [ -n "$COMP" ]; then
  oci iam compartment delete --compartment-id "$COMP" --force >/dev/null 2>&1 \
    && echo "compartment: deletion started (OCI finishes it in the background, often 10-60 min)" \
    || echo "compartment: FAILED to start deletion"
else
  echo "compartment: none"
fi
