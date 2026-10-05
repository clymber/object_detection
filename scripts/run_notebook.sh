#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <project>/notebooks/<file>.ipynb\n' "$0" >&2
    exit 2
fi
notebook="$1"

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
subprojects=("dataset" "ultralytics" "yolox" "rfdetr" "evaluation")

matches=()

if [[ "${notebook}" =~ ^nb[[:digit:].]{5,}$ ]]; then
    selector="$notebook"
    shopt -s nullglob
    for subproject in "${subprojects[@]}"; do
        matches+=("${workspace_root}/${subproject}/notebooks/${selector}-"*.ipynb)
    done
    shopt -u nullglob
else
    matches+=($(realpath ${workspace_root}/${notebook}))
fi

case "${#matches[@]}" in
    0)
        printf 'Notebook not found: %s\n' "${notebook}" >&2
        exit 2
        ;;
    1)
        if [[ ! -f "${matches[0]}" ]]; then
            printf "No such a file: %s\n" "${matches[0]}" >&2
            exit 2
        fi
        notebook="${matches[0]#"$workspace_root/"}"
        ;;
    *)
        printf 'Notebook selector is ambiguous: %s\n' "$selector" >&2
        for match in "${matches[@]}"; do
            printf '  %s\n' "${match#"$workspace_root/"}" >&2
        done
        exit 2
        ;;
esac

# Determine the project based on the notebook path.
case "$notebook" in
    dataset/notebooks/*.ipynb) project=dataset ;;
    ultralytics/notebooks/*.ipynb) project=ultralytics ;;
    yolox/notebooks/*.ipynb) project=yolox ;;
    rfdetr/notebooks/*.ipynb) project=rfdetr ;;
    evaluation/notebooks/*.ipynb) project=evaluation ;;
    *)
        printf 'Notebook must be under a project notebooks directory: %s\n' \
            "$notebook" >&2
        exit 2
        ;;
esac
project_env="object-detection-$project"


# Determine the conda binary based on the operating system.
conda_bin="${CONDA_EXE:-$(type -P conda || true)}"
if [[ "$(uname -s)" == Linux && ! -x "$conda_bin" ]]; then
    mountdir="${__MOUNTDIR__:-"${HOME}/work"}" # __MOUNTDIR__ from Renku
    conda_bin="${mountdir}/miniforge3/bin/conda"
fi
[[ -x "$conda_bin" ]] || {
    printf 'Conda not found. Set up Miniforge and notebook-tools first.\n' >&2
    exit 1
}

conda_cmd() {
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u PIP_PREFIX \
        -u PIP_TARGET -u PIP_USER "$conda_bin" "$@"
}

# Run the notebook using the determined project environment.
JUPYTER_PATH="$workspace_root/.conda/envs/$project_env/share/jupyter" \
    conda_cmd run -p "$workspace_root/.conda/envs/object-detection-notebooks" \
    jupyter nbconvert "$workspace_root/$notebook" \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="object-detection-$project" \
    --ExecutePreprocessor.timeout="${NOTEBOOK_TIMEOUT:--1}"
