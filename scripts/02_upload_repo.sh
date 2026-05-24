#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

require_cmd rsync
REMOTE_DIR="$(remote_workdir)"

remote_ssh "mkdir -p '${REMOTE_DIR}'"
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.env' \
  --exclude '.pytest_cache/' \
  --exclude 'results/' \
  --exclude 'hw1/results/' \
  --exclude 'hw2/results/' \
  --exclude 'hw3/results/' \
  -e "ssh -i ${SSH_PRIVATE_KEY_PATH} -p ${REMOTE_PORT:-22}" \
  "${PROJECT_ROOT}/" "$(ssh_target):${REMOTE_DIR}/"

echo "Uploaded repository to $(ssh_target):${REMOTE_DIR}"
