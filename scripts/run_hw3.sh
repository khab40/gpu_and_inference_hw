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
mkdir -p hw3/results
python -m pytest \
  hw3/test_cache_manager_correctness.py \
  hw3/test_scheduler_correctness.py \
  hw3/test_hw3_correctness.py \
  -q 2>&1 | tee hw3/results/hw3_tests.log
python hw3/hw3_task.py 2>&1 | tee hw3/results/hw3_run.log
'"
