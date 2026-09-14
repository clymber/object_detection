#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile=""
case "$(uname -s):$(uname -m)" in
    Darwin:arm64) profile="macos-mps"; conda_bin="${CONDA_EXE:-$(type -P conda || true)}" ;;
    Linux:x86_64) profile="linux-cuda"; conda_bin="/home/renku/work/miniforge3/bin/conda" ;;
    *) printf 'Unsupported host for the Conda profiles: %s %s\n' "$(uname -s)" "$(uname -m)" >&2; exit 2 ;;
esac
[[ -x "$conda_bin" ]] || {
    printf 'Conda not found at %s. On Renku run bash scripts/install_miniforge_renku.sh first.\n' "$conda_bin" >&2
    exit 1
}

# The Renku terminal starts inside a host venv. Never pass its Python settings
# to Conda or the selected project's interpreter.
conda_cmd() {
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u PIP_PREFIX \
        -u PIP_TARGET -u PIP_USER "$conda_bin" "$@"
}

usage() {
    printf 'Usage: %s [all|detection_common|evaluation|dataset|rfdetr|ultralytics|yolox|notebook-tools]...\n' "$0"
}

environment_name() {
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

yolox_build_dir=""
cleanup_yolox_build() {
    if [[ "$yolox_build_dir" == /tmp/object-detection-yolox.* && -d "$yolox_build_dir" ]]; then
        rm -rf -- "$yolox_build_dir"
    fi
}
trap cleanup_yolox_build EXIT

install_yolox_linux() {
    local env_name="$1"
    local commit="6ddff4824372906469a7fae2dc3206c7aa4bbaee"
    if conda_cmd run -n "$env_name" python -c \
        "import yolox.layers.fast_cocoeval; from yolox._object_ctrl_build import SOURCE_COMMIT; assert SOURCE_COMMIT == '$commit'" \
        >/dev/null 2>&1; then
        return
    fi
    command -v git >/dev/null || { printf 'git is required to install YOLOX.\n' >&2; return 1; }
    command -v c++ >/dev/null || { printf 'A C++ compiler is required to install YOLOX.\n' >&2; return 1; }
    yolox_build_dir="$(mktemp -d /tmp/object-detection-yolox.XXXXXX)"
    git -C "$yolox_build_dir" init --quiet
    git -C "$yolox_build_dir" remote add origin https://github.com/Megvii-BaseDetection/YOLOX.git
    git -C "$yolox_build_dir" fetch --depth 1 origin "$commit"
    git -C "$yolox_build_dir" checkout --quiet --detach FETCH_HEAD
    [[ "$(git -C "$yolox_build_dir" rev-parse HEAD)" == "$commit" ]] || {
        printf 'YOLOX checkout is not at the expected commit.\n' >&2
        return 1
    }
    python3 - "$yolox_build_dir" "$commit" <<'PY'
import sys
from pathlib import Path

checkout = Path(sys.argv[1])
commit = sys.argv[2]
jit_path = checkout / "yolox/layers/jit_ops.py"
requirements_path = checkout / "requirements.txt"
jit = jit_path.read_text()
requirements = requirements_path.read_text()
if jit.count("-std=c++14") != 2:
    raise SystemExit("Unexpected YOLOX compiler flag layout")
if requirements.splitlines().count("opencv_python") != 1:
    raise SystemExit("Unexpected YOLOX OpenCV requirement")
if requirements.splitlines().count("onnx-simplifier==0.4.10") != 1:
    raise SystemExit("Unexpected YOLOX ONNX requirement")
jit_path.write_text(jit.replace("-std=c++14", "-std=c++17"))
requirements_path.write_text(
    requirements.replace("opencv_python", "opencv-python-headless")
    .replace("onnx-simplifier==0.4.10", "onnx-simplifier")
)
(checkout / "yolox/_object_ctrl_build.py").write_text(
    f'SOURCE_COMMIT = "{commit}"\nCXX_STANDARD = "c++17"\n'
)
PY
    conda_cmd run -n "$env_name" python -m pip install \
        --no-build-isolation --no-deps --force-reinstall "$yolox_build_dir"
    conda_cmd run -n "$env_name" python -c \
        "import yolox.layers.fast_cocoeval; from yolox._object_ctrl_build import SOURCE_COMMIT; assert SOURCE_COMMIT == '$commit'"
    cleanup_yolox_build
    yolox_build_dir=""
}

validate_linux_gpu() {
    local env_name="$1"
    local project="$2"
    conda_cmd run -n "$env_name" python -c '
import importlib.metadata as metadata
import cv2
import torch
import torchvision
from torchvision.ops import nms
if not torch.version.cuda or not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch cannot access the allocated GPU")
device = torch.device("cuda:0")
assert torch.tensor([1., 2.], device=device).square().sum().item() == 5.
boxes = torch.tensor([[0., 0., 4., 4.]], device=device)
assert nms(boxes, torch.tensor([0.9], device=device), 0.5).numel() == 1
opencv_names = {"opencv-python", "opencv-python-headless",
                "opencv-contrib-python", "opencv-contrib-python-headless"}
installed = {d.metadata["Name"].lower() for d in metadata.distributions()}
assert installed & opencv_names == {"opencv-python-headless"}, installed & opencv_names
gui = [line.split(":", 1)[1].strip() for line in cv2.getBuildInformation().splitlines()
       if line.strip().startswith("GUI:")]
assert gui == ["NONE"], gui
print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
'
    case "$project" in
        ultralytics)
            conda_cmd run -n "$env_name" python -c 'import ultralytics'
            ;;
        yolox)
            conda_cmd run -n "$env_name" python -c 'import yolox.layers.fast_cocoeval'
            ;;
        rfdetr)
            conda_cmd run -n "$env_name" python -c 'from rfdetr import RFDETRSmall'
            ;;
    esac
}

setup_one() {
    local project="$1"
    local env_name
    env_name="$(environment_name "$project")" || { usage >&2; return 2; }
    local definition="$workspace_root/$project/environment-$profile.yml"
    [[ -f "$definition" ]] || { printf 'Missing manifest: %s\n' "$definition" >&2; return 1; }

    local manifest_hash
    manifest_hash="$(cksum "$definition" | awk '{print $1 ":" $2}')"
    local prefix=""
    if conda_cmd run -n "$env_name" python -c 'pass' >/dev/null 2>&1; then
        prefix="$(conda_cmd run -n "$env_name" python -c 'import sys; print(sys.prefix)' | tail -1)"
        if [[ ! -f "$prefix/.object-detection-manifest" ]] ||
            [[ "$(cat "$prefix/.object-detection-manifest")" != "$manifest_hash" ]]; then
            conda_cmd env update -f "$definition"
        fi
    else
        conda_cmd env create -f "$definition"
    fi
    prefix="$(conda_cmd run -n "$env_name" python -c 'import sys; print(sys.prefix)' | tail -1)"

    if [[ "$project" == notebook-tools ]]; then
        conda_cmd run -n "$env_name" python -m pip check
        printf '%s\n' "$manifest_hash" > "$prefix/.object-detection-manifest"
        du -sh "$prefix"
        return
    fi

    if [[ "$project" == yolox ]]; then
        if [[ "$profile" == linux-cuda ]]; then
            install_yolox_linux "$env_name"
        else
            local source_dir="${YOLOX_SOURCE:-$workspace_root/../YOLOX}"
            [[ "$source_dir" == /* && -f "$source_dir/setup.py" ]] || {
                printf 'Set YOLOX_SOURCE to an absolute YOLOX checkout path.\n' >&2
                return 1
            }
            if ! conda_cmd run -n "$env_name" python -c \
                'import yolox.data, yolox.layers.fast_cocoeval' >/dev/null 2>&1; then
                conda_cmd run -n "$env_name" python -m pip install \
                    --no-build-isolation --no-deps -e "$source_dir"
            fi
        fi
    fi

    local installs=("$workspace_root/detection_common")
    case "$project" in
        detection_common) ;;
        evaluation) installs+=("$workspace_root/evaluation") ;;
        dataset) installs+=("$workspace_root/dataset") ;;
        rfdetr) installs+=("$workspace_root/evaluation" "$workspace_root/dataset" "$workspace_root/rfdetr") ;;
        ultralytics) installs+=("$workspace_root/evaluation" "$workspace_root/ultralytics") ;;
        yolox) installs+=("$workspace_root/evaluation" "$workspace_root/yolox") ;;
    esac
    local paths=()
    local path
    for path in "${installs[@]}"; do paths+=(-e "$path"); done
    conda_cmd run -n "$env_name" python -m pip install --no-deps "${paths[@]}"

    if [[ "$profile" == linux-cuda ]]; then
        if [[ "$project" == rfdetr || "$project" == ultralytics ||
              "$project" == yolox ]]; then
            validate_linux_gpu "$env_name" "$project"
            if [[ "$project" == rfdetr ]]; then
                conda_cmd run -n "$env_name" python \
                    "$workspace_root/scripts/check_renku_rfdetr.py"
            fi
        fi
        conda_cmd run -n "$env_name" python -m pip check
    elif [[ "$project" != yolox ]]; then
        conda_cmd run -n "$env_name" python -m pip check
    fi

    if [[ "$project" != detection_common ]]; then
        local kernel="object-detection-$project"
        local display="Object Detection $project ($profile)"
        conda_cmd run -n "$env_name" python -m ipykernel install \
            --user --name "$kernel" --display-name "$display" \
            --env PYTHONPATH "" --env PYTHONNOUSERSITE "1"
    fi
    printf '%s\n' "$manifest_hash" > "$prefix/.object-detection-manifest"
    du -sh "$prefix"
}

if [[ $# -eq 0 || "$1" == all ]]; then
    set -- detection_common evaluation dataset rfdetr ultralytics yolox notebook-tools
fi
for project in "$@"; do setup_one "$project"; done
