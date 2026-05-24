#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_hw1.sh"
"${SCRIPT_DIR}/run_hw2.sh"
"${SCRIPT_DIR}/run_hw3.sh"
