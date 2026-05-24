#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

require_cmd nebius
require_var NEBIUS_VM_ID

nebius compute instance stop "${NEBIUS_VM_ID}"
echo "Stopped VM ${NEBIUS_VM_ID}"
