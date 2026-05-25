#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

REMOTE_DIR="$(remote_workdir)"
HW2_CUSTOM_KV_ARGS="${HW2_CUSTOM_KV_ARGS:---include-baseline --profile}"

remote_ssh "bash -lc '
set -euo pipefail
cd \"${REMOTE_DIR}\"
if ! python3 - <<\"PY\" >/dev/null 2>&1
import sysconfig
from pathlib import Path
raise SystemExit(0 if (Path(sysconfig.get_paths()[\"include\"]) / \"Python.h\").exists() else 1)
PY
then
  sudo apt-get update
  sudo apt-get install -y python3-dev build-essential
fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p hw2/results
python - <<\"PY\"
import torch
print(\"CUDA available:\", torch.cuda.is_available())
print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")
PY
python hw2/hw2-custom-kv.py ${HW2_CUSTOM_KV_ARGS} 2>&1 | tee hw2/results/hw2_custom_kv_run.log
'"
