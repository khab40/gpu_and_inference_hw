#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

require_cmd rsync
REMOTE_DIR="$(remote_workdir)"
RUN_DIR="$(local_results_dir)"
mkdir -p "${RUN_DIR}/hw1" "${RUN_DIR}/hw2" "${RUN_DIR}/hw3"

for hw in hw1 hw2 hw3; do
  rsync -az \
    -e "ssh -i ${SSH_PRIVATE_KEY_PATH} -p ${REMOTE_PORT:-22}" \
    "$(ssh_target):${REMOTE_DIR}/${hw}/results/" \
    "${RUN_DIR}/${hw}/"
done

echo "Collected results into ${RUN_DIR}"
