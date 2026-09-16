#!/usr/bin/env bash

set -euo pipefail

readonly bashrc="${HOME}/.bashrc"

touch "${bashrc}"

append_if_missing() {
    local marker="$1"
    local content="$2"

    if ! grep -Fq -- "${marker}" "${bashrc}"; then
        printf '\n%s\n' "${content}" >> "${bashrc}"
    fi
}

conda init bash
# conda config --set auto_activate false
conda config --set changeps1 false

append_if_missing \
    'source "$HOME/.bash_prompt_rc"' \
    'source "$HOME/.bash_prompt_rc"'

append_if_missing \
    'source "$HOME/.config/roboflow.rc"' \
    'if [ -f "$HOME/.config/roboflow.rc" ]; then
        source "$HOME/.config/roboflow.rc"
    fi'
