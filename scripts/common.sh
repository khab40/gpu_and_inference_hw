#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

load_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: ${name}. Set it in ${ENV_FILE} or export it." >&2
    exit 1
  fi
}

set_env_var() {
  local key="$1"
  local value="$2"
  touch "${ENV_FILE}"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    python3 - "$ENV_FILE" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text().splitlines()
prefix = f"{key}="
for i, line in enumerate(lines):
    if line.startswith(prefix):
        lines[i] = f"{key}={value}"
        break
path.write_text("\n".join(lines) + "\n")
PY
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
  export "${key}=${value}"
}

ssh_target() {
  require_var REMOTE_USER
  require_var REMOTE_HOST
  printf '%s@%s' "${REMOTE_USER}" "${REMOTE_HOST}"
}

ssh_base_args() {
  require_var SSH_PRIVATE_KEY_PATH
  local port="${REMOTE_PORT:-22}"
  printf '%s\n' -i "${SSH_PRIVATE_KEY_PATH}" -p "${port}" -o StrictHostKeyChecking=accept-new
}

remote_ssh() {
  require_var SSH_PRIVATE_KEY_PATH
  ssh -i "${SSH_PRIVATE_KEY_PATH}" \
    -p "${REMOTE_PORT:-22}" \
    -o StrictHostKeyChecking=accept-new \
    "$(ssh_target)" \
    "$@"
}

remote_workdir() {
  printf '%s' "${REMOTE_WORKDIR:-gpu_and_inference_hw}"
}

local_results_dir() {
  printf '%s' "${LOCAL_RESULTS_DIR:-${PROJECT_ROOT}/results/${VM_NAME:-nebius-run}}"
}
