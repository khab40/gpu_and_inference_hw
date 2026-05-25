#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

require_cmd rsync
REMOTE_DIR="$(remote_workdir)"
RUN_DIR="$(local_results_dir)"
mkdir -p "${RUN_DIR}"

collected=0
for hw in hw1 hw2 hw3; do
  if ! remote_ssh "test -d '${REMOTE_DIR}/${hw}/results'"; then
    echo "Skipping ${hw}: no remote results directory at ${REMOTE_DIR}/${hw}/results"
    continue
  fi

  mkdir -p "${RUN_DIR}/${hw}"
  rsync -az \
    -e "ssh -i ${SSH_PRIVATE_KEY_PATH} -p ${REMOTE_PORT:-22}" \
    "$(ssh_target):${REMOTE_DIR}/${hw}/results/" \
    "${RUN_DIR}/${hw}/"
  collected=$((collected + 1))
done

if [[ "${collected}" -eq 0 ]]; then
  echo "No homework results were found under ${REMOTE_DIR}."
  exit 0
fi

echo "Collected results into ${RUN_DIR}"
