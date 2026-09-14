#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <project>/notebooks/<file>.ipynb\n' "$0" >&2
    exit 2
fi

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
notebook="$1"
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
if [[ ! -f "$workspace_root/$notebook" ]]; then
    printf 'Notebook not found: %s\n' "$notebook" >&2
    exit 2
fi

if [[ "$(uname -s)" == Linux ]]; then
    conda_bin="/home/renku/work/miniforge3/bin/conda"
else
    conda_bin="${CONDA_EXE:-$(type -P conda || true)}"
fi
[[ -x "$conda_bin" ]] || {
    printf 'Conda not found. Set up Miniforge and notebook-tools first.\n' >&2
    exit 1
}
env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u PIP_PREFIX \
    -u PIP_TARGET -u PIP_USER "$conda_bin" run -n notebook-tools \
    jupyter nbconvert "$workspace_root/$notebook" \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="object-detection-$project" \
    --ExecutePreprocessor.timeout="${NOTEBOOK_TIMEOUT:--1}"
