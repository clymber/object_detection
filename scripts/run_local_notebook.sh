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

conda run -n notebook-tools jupyter nbconvert "$workspace_root/$notebook" \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="object-detection-$project" \
    --ExecutePreprocessor.timeout="${NOTEBOOK_TIMEOUT:--1}"
