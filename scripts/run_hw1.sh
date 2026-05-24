#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

REMOTE_DIR="$(remote_workdir)"
remote_ssh "bash -lc '
set -euo pipefail
cd \"${REMOTE_DIR}\"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p hw1/results
python - <<\"PY\"
import torch
print(\"CUDA available:\", torch.cuda.is_available())
print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")
PY
python hw1/hw1_task.py 2>&1 | tee hw1/results/hw1_run.log
'"
