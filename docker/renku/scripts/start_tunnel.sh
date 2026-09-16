#!/bin/bash

set -e

RENKU_WORKING_DIR="${RENKU_WORKING_DIR:-${HOME}}"
RENKU_MOUNT_DIR="${RENKU_MOUNT_DIR:-${RENKU_WORKING_DIR}}"
CODE_CLI_PATH="${CODE_CLI_PATH:-/home/vscode/.local/bin/code}"
VSCODE_EXTENSIONS_DIR="${VSCODE_EXTENSIONS_DIR:-/home/vscode/.vscode_tunnel/extensions}"

"${CODE_CLI_PATH}" --version

# Generate stable tunnel name if possible
if [ -n "${RENKU_BASE_URL_PATH}" ]; then
    TUNNEL_NAME=$(echo -n "${RENKU_BASE_URL_PATH}" | md5sum)
elif [ -n "${HOSTNAME}" ]; then
    TUNNEL_NAME=$(echo -n "${HOSTNAME}" | md5sum)
else
    echo "Warning: could not find session name or hostname, using random string"
    TUNNEL_NAME=$(LC_ALL=C tr -dc "a-z0-9" </dev/urandom 2>/dev/null | head -c 20)
fi
TUNNEL_NAME=$(echo "renku-${TUNNEL_NAME}" | cut -c 1-20)
echo "TUNNEL_NAME: ${TUNNEL_NAME}"

"${CODE_CLI_PATH}" tunnel \
    --name "${TUNNEL_NAME}" \
    --server-data-dir "${RENKU_MOUNT_DIR}/.vscode_tunnel" \
    --extensions-dir "${VSCODE_EXTENSIONS_DIR}" \
    --cli-data-dir "${RENKU_MOUNT_DIR}/.vscode_tunnel/cli"
