#!/usr/bin/env bash

set -euo pipefail

readonly code_cli_path="${CODE_CLI_PATH:-${HOME}/.local/bin/code}"
readonly extensions_dir="${VSCODE_EXTENSIONS_DIR:-${HOME}/.vscode_tunnel/extensions}"
readonly code_cli_url="https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"

readonly tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

mkdir -p "$(dirname "${code_cli_path}")" "${extensions_dir}"

curl -fsSL "${code_cli_url}" -o "${tmp_dir}/code_cli.tar.gz"
tar -xzf "${tmp_dir}/code_cli.tar.gz" -C "${tmp_dir}"
install -m 0755 "${tmp_dir}/code" "${code_cli_path}"

vscode_commit="$("${code_cli_path}" --version \
    | sed -n 's/.*commit \([0-9a-f]\{40\}\).*/\1/p')"
readonly vscode_commit
if [[ -z "${vscode_commit}" ]]; then
    echo "Could not determine the VS Code commit from ${code_cli_path}" >&2
    exit 1
fi

# Use a temporary server matching the CLI commit so the extension manager
# selects packages compatible with the tunnel server.
curl -fsSL \
    "https://update.code.visualstudio.com/commit:${vscode_commit}/server-linux-x64/stable" \
    -o "${tmp_dir}/vscode_server.tar.gz"
mkdir "${tmp_dir}/server"
tar -xzf "${tmp_dir}/vscode_server.tar.gz" \
    --strip-components=1 -C "${tmp_dir}/server"

# Rewrap's Marketplace signature metadata is rejected by the VS Code CLI, so
# install its latest stable Open VSX package as a local VSIX instead.
rewrap_url="$(curl -fsSL 'https://open-vsx.org/api/stkb/rewrap/latest' \
    | jq -er '.files.download')"
readonly rewrap_url
curl -fsSL "${rewrap_url}" -o "${tmp_dir}/stkb.rewrap.vsix"

readonly extensions=(
    openai.chatgpt
    ReprEng.csv
    ZainChen.json
    ms-toolsai.jupyter
    ms-toolsai.vscode-jupyter-cell-tags
    ms-toolsai.jupyter-renderers
    ms-toolsai.vscode-jupyter-slideshow
    ms-vscode.makefile-tools
    DavidAnson.vscode-markdownlint
    ms-python.python
    ms-python.vscode-pylance
    ms-python.debugpy
    ms-python.vscode-python-envs
    charliermarsh.ruff
    caenrigen.jupytext-sync
)

install_args=()
for extension in "${extensions[@]}"; do
    install_args+=(--install-extension "${extension}")
done
readonly install_args

"${tmp_dir}/server/bin/code-server" \
    --extensions-dir "${extensions_dir}" \
    "${install_args[@]}" \
    --install-extension "${tmp_dir}/stkb.rewrap.vsix"

"${tmp_dir}/server/bin/code-server" \
    --extensions-dir "${extensions_dir}" \
    --list-extensions --show-versions
