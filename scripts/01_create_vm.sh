#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_env

require_cmd nebius
require_cmd jq
require_cmd ssh-keygen
require_cmd ssh
require_var NEBIUS_PROJECT_ID

VM_NAME="${VM_NAME:-gpu-inference-hw-$(date +%Y%m%d-%H%M%S)}"
REMOTE_USER="${REMOTE_USER:-user}"
SSH_PRIVATE_KEY_PATH="${SSH_PRIVATE_KEY_PATH:-${HOME}/.ssh/${VM_NAME}_ed25519}"
SSH_PUBLIC_KEY_PATH="${SSH_PUBLIC_KEY_PATH:-${SSH_PRIVATE_KEY_PATH}.pub}"
PLATFORM="${NEBIUS_PLATFORM:-gpu-l40s-a}"
PRESET="${NEBIUS_PRESET:-1gpu-16vcpu-64gb}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-200}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-network_ssd}"
IMAGE_FAMILY="${NEBIUS_IMAGE_FAMILY:-ubuntu24.04-cuda13.0}"

mkdir -p "$(dirname "${SSH_PRIVATE_KEY_PATH}")"
if [[ ! -f "${SSH_PRIVATE_KEY_PATH}" ]]; then
  ssh-keygen -t ed25519 -f "${SSH_PRIVATE_KEY_PATH}" -C "${VM_NAME}" -N ""
fi

nebius config set parent-id "${NEBIUS_PROJECT_ID}" >/dev/null

SUBNET_ID="${NEBIUS_SUBNET_ID:-}"
if [[ -z "${SUBNET_ID}" ]]; then
  SUBNET_ID="$(nebius vpc subnet list --format json | jq -r '.items[0].metadata.id')"
fi
if [[ -z "${SUBNET_ID}" || "${SUBNET_ID}" == "null" ]]; then
  echo "Could not discover subnet. Set NEBIUS_SUBNET_ID in .env." >&2
  exit 1
fi

BOOT_DISK_ID="$(nebius compute disk create \
  --name "${VM_NAME}-boot" \
  --size-gibibytes "${BOOT_DISK_SIZE_GB}" \
  --type "${BOOT_DISK_TYPE}" \
  --source-image-family-image-family "${IMAGE_FAMILY}" \
  --block-size-bytes 4096 \
  --format json | jq -r '.metadata.id')"

USER_DATA="$(jq -Rs '.' <<EOF
users:
  - name: ${REMOTE_USER}
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "${SSH_PUBLIC_KEY_PATH}")
EOF
)"

VM_ID="$(nebius compute instance create \
  --format json \
  - <<EOF | jq -r '.metadata.id'
{
  "metadata": { "name": "${VM_NAME}" },
  "spec": {
    "stopped": false,
    "cloud_init_user_data": ${USER_DATA},
    "resources": {
      "platform": "${PLATFORM}",
      "preset": "${PRESET}"
    },
    "boot_disk": {
      "attach_mode": "READ_WRITE",
      "existing_disk": { "id": "${BOOT_DISK_ID}" }
    },
    "network_interfaces": [{
      "name": "${VM_NAME}-nic",
      "subnet_id": "${SUBNET_ID}",
      "ip_address": {},
      "public_ip_address": {}
    }]
  }
}
EOF
)"

for _ in $(seq 1 60); do
  REMOTE_HOST="$(nebius compute instance get \
    --id "${VM_ID}" \
    --format json | jq -r '.status.network_interfaces[0].public_ip_address.address // empty' | cut -d/ -f1)"
  [[ -n "${REMOTE_HOST}" ]] && break
  sleep 10
done

if [[ -z "${REMOTE_HOST:-}" ]]; then
  echo "VM was created, but no public IP was reported. VM_ID=${VM_ID}" >&2
  exit 1
fi

set_env_var VM_NAME "${VM_NAME}"
set_env_var NEBIUS_PLATFORM "${PLATFORM}"
set_env_var NEBIUS_PRESET "${PRESET}"
set_env_var NEBIUS_SUBNET_ID "${SUBNET_ID}"
set_env_var NEBIUS_VM_ID "${VM_ID}"
set_env_var NEBIUS_BOOT_DISK_ID "${BOOT_DISK_ID}"
set_env_var REMOTE_USER "${REMOTE_USER}"
set_env_var REMOTE_HOST "${REMOTE_HOST}"
set_env_var REMOTE_PORT "${REMOTE_PORT:-22}"
set_env_var SSH_PRIVATE_KEY_PATH "${SSH_PRIVATE_KEY_PATH}"
set_env_var SSH_PUBLIC_KEY_PATH "${SSH_PUBLIC_KEY_PATH}"
set_env_var REMOTE_WORKDIR "${REMOTE_WORKDIR:-gpu_and_inference_hw}"

echo "VM created: ${VM_ID}"
echo "Boot disk:  ${BOOT_DISK_ID}"
echo "SSH host:   ${REMOTE_USER}@${REMOTE_HOST}"
ssh $(ssh_base_args) "$(ssh_target)" "nvidia-smi"
