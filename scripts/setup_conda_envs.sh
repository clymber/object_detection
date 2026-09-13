#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    # Explain selective local setup.
    printf 'Usage: %s [all|detection_common|evaluation|dataset|rfdetr|ultralytics|yolox|notebook-tools]...\n' "$0"
}

environment_name() {
    # Return the declared Conda name for a project directory.
    case "$1" in
        detection_common) printf 'detection-common-dev' ;;
        evaluation) printf 'evaluation-dev' ;;
        dataset) printf 'dataset-dev' ;;
        rfdetr) printf 'rfdetr-dev' ;;
        ultralytics) printf 'ultralytics-dev' ;;
        yolox) printf 'yolox-dev' ;;
        notebook-tools) printf 'notebook-tools' ;;
        *) return 2 ;;
    esac
}

setup_one() {
    # Create/update one environment, then install its internal editable packages.
    local project="$1"
    local env_name
    env_name="$(environment_name "$project")" || {
        usage >&2
        return 2
    }
    local definition="$workspace_root/$project/environment.yml"
    if conda run -n "$env_name" python -c 'pass' >/dev/null 2>&1; then
        conda env update -f "$definition"
    else
        conda env create -f "$definition"
    fi
    if [[ "$project" == notebook-tools ]]; then
        return
    fi

    local installs=("$workspace_root/detection_common")
    case "$project" in
        evaluation) installs+=("$workspace_root/evaluation") ;;
        dataset) installs+=("$workspace_root/dataset") ;;
        rfdetr)
            installs+=(
                "$workspace_root/evaluation"
                "$workspace_root/dataset"
                "$workspace_root/rfdetr"
            )
            ;;
        ultralytics)
            installs+=("$workspace_root/evaluation" "$workspace_root/ultralytics")
            ;;
        yolox)
            local source_dir="${YOLOX_SOURCE:-$workspace_root/../YOLOX}"
            if [[ "$source_dir" != /* ]]; then
                printf 'YOLOX_SOURCE must be an absolute path: %s\n' \
                    "$source_dir" >&2
                return 2
            fi
            if [[ ! -f "$source_dir/setup.py" ]]; then
                printf 'YOLOX checkout not found: %s\n' "$source_dir" >&2
                return 1
            fi
            if ! conda run -n "$env_name" python -c \
                'import yolox.data, yolox.layers.fast_cocoeval' >/dev/null 2>&1; then
                conda run -n "$env_name" python -m pip install \
                    --no-build-isolation --no-deps -e "$source_dir"
            fi
            installs+=("$workspace_root/evaluation" "$workspace_root/yolox")
            ;;
    esac
    local package_paths=()
    local path
    for path in "${installs[@]}"; do
        package_paths+=(-e "$path")
    done
    conda run -n "$env_name" python -m pip install --no-deps "${package_paths[@]}"

    if [[ "$project" != detection_common ]]; then
        local display="Object Detection ${project} (local)"
        conda run -n "$env_name" python -m ipykernel install \
            --user --name "object-detection-${project}" --display-name "$display"
    fi
}

if [[ $# -eq 0 || "$1" == all ]]; then
    set -- detection_common evaluation dataset rfdetr ultralytics yolox notebook-tools
fi

for project in "$@"; do
    setup_one "$project"
done
