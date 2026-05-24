#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

require_cmd nebius
require_var NEBIUS_VM_ID
require_var NEBIUS_BOOT_DISK_ID

echo "Deleting VM ${NEBIUS_VM_ID}"
nebius compute instance delete "${NEBIUS_VM_ID}"

echo "Deleting boot disk ${NEBIUS_BOOT_DISK_ID}"
nebius compute disk delete "${NEBIUS_BOOT_DISK_ID}"

echo "Deleted VM and boot disk."
