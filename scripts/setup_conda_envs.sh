#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

readonly subprojects=(
    "detection_common"
    "evaluation"
    "dataset"
    "rfdetr"
    "ultralytics"
    "yolox"
    "notebook-tools"
)

usage() {
    printf "Usage: %s [subproject]...\n" "$0"
    printf "\tAvailable subprojects:\n"
    for subproject in "all" "${subprojects[@]}"; do
        printf "\t\t%s\n" "$subproject"
    done
}

# Keep environment names stable across differently named workspace checkouts.
readonly workspace_id="object-detection"

# Conda profile and binary
profile=""
conda_bin="${CONDA_EXE:-$(type -P conda || true)}"
case "$(uname -s):$(uname -m)" in
    Darwin:arm64)
        profile="macos-mps" ;;
    Linux:x86_64)
        profile="linux-cuda"
        if [[ ! -x "$conda_bin" ]]; then
            mountdir="${__MOUNTDIR__:-"${HOME}/work"}" # __MOUNTDIR__ from Renku
            conda_bin="${mountdir}/miniforge3/bin/conda"
        fi
        ;;
    *)
        printf 'Unsupported host for the Conda profiles: %s %s\n' "$(uname -s)" "$(uname -m)" >&2; exit 2 ;;
esac
[[ -x "$conda_bin" ]] || {
    printf 'Conda not found at %s. Please install Conda first.\n' "$conda_bin" >&2
    exit 1
}
command -v jq >/dev/null || {
    printf 'jq is required to inspect the Conda configuration.\n' >&2
    exit 1
}

# The Renku terminal may start inside a host venv. Clear the previous Python
# environment settings if they exist.
conda_cmd() {
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u PIP_PREFIX \
        -u PIP_TARGET -u PIP_USER "$conda_bin" "$@"
}

# Get the environment name for a given subproject.
environment_name() {
    local subproject="$1"

    case "${subproject}" in
        detection_common)
            printf '%s-common' "$workspace_id" ;;
        notebook-tools)
            printf '%s-notebooks' "$workspace_id" ;;
        evaluation|dataset|rfdetr|ultralytics|yolox)
            printf '%s-%s' "$workspace_id" "$subproject" ;;
        *)
            return 2 ;;
    esac
}

# Register the Conda environments directory if it is not already registered.
register_envs_dir() {
    local envs_dir="$1"

    if conda_cmd config --show envs_dirs --json |
        jq -e --arg dir "$envs_dir" '.envs_dirs | index($dir) != null' >/dev/null
    then
        return 0
    fi

    conda_cmd config --append envs_dirs "$envs_dir"
}

# YOLOX build directory and cleanup function.
yolox_build_dir=""
cleanup_yolox_build() {
    if [[ "$yolox_build_dir" == /tmp/object-detection-yolox.* && -d "$yolox_build_dir" ]]; then
        rm -rf -- "$yolox_build_dir"
    fi
}
trap cleanup_yolox_build EXIT

# Install YOLOX in a Conda environment on Linux.
install_yolox_linux() {
    local env_prefix="$1"
    local yolox_url="https://github.com/Megvii-BaseDetection/YOLOX.git"
    local commit="6ddff4824372906469a7fae2dc3206c7aa4bbaee"

    local check_commit=$(printf '%s\n' \
        "import yolox.layers.fast_cocoeval" \
        "from yolox._object_ctrl_build import SOURCE_COMMIT" \
        "assert SOURCE_COMMIT == '$commit'"
    )

    if conda_cmd run -p "$env_prefix" \
        python -c "${check_commit}" >/dev/null 2>&1
    then
        return 0
    fi
    if ! command -v git >/dev/null; then
        printf 'git is required to install YOLOX.\n' >&2
        return 1
    fi
    if ! command -v c++ >/dev/null; then
        printf 'A C++ compiler is required to install YOLOX.\n' >&2
        return 1
    fi

    yolox_build_dir="$(mktemp -d /tmp/object-detection-yolox.XXXXXX)"
    git -C "$yolox_build_dir" init --quiet
    git -C "$yolox_build_dir" remote add origin "$yolox_url"
    git -C "$yolox_build_dir" fetch --depth 1 origin "$commit"
    git -C "$yolox_build_dir" checkout --quiet --detach FETCH_HEAD
    [[ "$(git -C "$yolox_build_dir" rev-parse HEAD)" == "$commit" ]] || {
        printf 'YOLOX checkout is not at the expected commit.\n' >&2
        return 1
    }
    "$env_prefix/bin/python" - "$yolox_build_dir" "$commit" <<-'PY'
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

    conda_cmd run -p "$env_prefix" python -m pip install \
        --no-build-isolation --no-deps --force-reinstall "$yolox_build_dir"
    conda_cmd run -p "$env_prefix" python -c "${check_commit}"
    cleanup_yolox_build
    yolox_build_dir=""
}

validate_linux_gpu() {
    local env_prefix="$1"
    local project="$2"
    local validation
    validation="$(printf '%s\n' \
        'import importlib.metadata as metadata' \
        'import cv2' \
        'import torch' \
        'import torchvision' \
        'from torchvision.ops import nms' \
        'if not torch.version.cuda or not torch.cuda.is_available():' \
        '    raise SystemExit("CUDA-enabled PyTorch cannot access the allocated GPU")' \
        'device = torch.device("cuda:0")' \
        'assert torch.tensor([1., 2.], device=device).square().sum().item() == 5.' \
        'boxes = torch.tensor([[0., 0., 4., 4.]], device=device)' \
        'assert nms(boxes, torch.tensor([0.9], device=device), 0.5).numel() == 1' \
        'opencv_names = {"opencv-python", "opencv-python-headless",' \
        '                "opencv-contrib-python", "opencv-contrib-python-headless"}' \
        'installed = {d.metadata["Name"].lower() for d in metadata.distributions()}' \
        'assert installed & opencv_names == {"opencv-python-headless"}, installed & opencv_names' \
        'gui = [line.split(":", 1)[1].strip() for line in cv2.getBuildInformation().splitlines()' \
        '       if line.strip().startswith("GUI:")]' \
        'assert gui == ["NONE"], gui' \
        'print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' \
    )"
    conda_cmd run -p "$env_prefix" python -c "$validation"

    case "$project" in
        ultralytics)
            conda_cmd run -p "$env_prefix" python -c 'import ultralytics'
            ;;
        yolox)
            conda_cmd run -p "$env_prefix" python -c 'import yolox.layers.fast_cocoeval'
            ;;
        rfdetr)
            conda_cmd run -p "$env_prefix" python -c 'from rfdetr import RFDETRSmall'
            ;;
    esac
}

setup_one() {
    local project="$1"
    local env_name
    env_name="$(environment_name "$project")" || { usage >&2; return 2; }
    local env_prefix="$workspace_root/.conda/envs/$env_name"
    register_envs_dir "${env_prefix%/*}"
    local definition="$workspace_root/$project/environment-$profile.yml"
    [[ -f "$definition" ]] || { printf 'Missing manifest: %s\n' "$definition" >&2; return 1; }

    local manifest_hash
    manifest_hash="$(cksum "$definition" | awk '{print $1 ":" $2}')"
    local prefix=""
    if conda_cmd run -p "$env_prefix" python -c 'pass' >/dev/null 2>&1; then
        prefix="$(conda_cmd run -p "$env_prefix" python -c 'import sys; print(sys.prefix)' | tail -1)"
        if [[ ! -f "$prefix/.object-detection-manifest" ]] ||
            [[ "$(cat "$prefix/.object-detection-manifest")" != "$manifest_hash" ]]; then
            conda_cmd env update -p "$env_prefix" -f "$definition"
        fi
    else
        conda_cmd env create -p "$env_prefix" -f "$definition"
    fi
    prefix="$(conda_cmd run -p "$env_prefix" python -c 'import sys; print(sys.prefix)' | tail -1)"
    if [[ "$project" == notebook-tools ]]; then
        conda_cmd run -p "$env_prefix" python -m pip check
        printf '%s\n' "$manifest_hash" > "$prefix/.object-detection-manifest"
        du -sh "$prefix"
        return
    fi

    if [[ "$project" == yolox ]]; then
        if [[ "$profile" == linux-cuda ]]; then
            install_yolox_linux "$env_prefix"
        else
            local source_dir="${YOLOX_SOURCE:-$workspace_root/../YOLOX}"
            [[ "$source_dir" == /* && -f "$source_dir/setup.py" ]] || {
                printf 'Set YOLOX_SOURCE to an absolute YOLOX checkout path.\n' >&2
                return 1
            }
            if ! conda_cmd run -p "$env_prefix" python -c \
                'import yolox.data, yolox.layers.fast_cocoeval' >/dev/null 2>&1; then
                conda_cmd run -p "$env_prefix" python -m pip install \
                    --no-build-isolation --no-deps -e "$source_dir"
            fi
        fi
    fi

    local installs=("$workspace_root/detection_common")
    case "$project" in
        detection_common)
			;;
        evaluation)
			installs+=("$workspace_root/evaluation") ;;
        dataset)
			installs+=("$workspace_root/dataset") ;;
        rfdetr)
			installs+=("$workspace_root/evaluation" "$workspace_root/dataset" "$workspace_root/rfdetr") ;;
        ultralytics)
            installs+=("$workspace_root/evaluation" "$workspace_root/dataset" "$workspace_root/ultralytics") ;;
        yolox)
			installs+=("$workspace_root/evaluation" "$workspace_root/dataset" "$workspace_root/yolox") ;;
    esac

    local paths=()
    local path
    for path in "${installs[@]}"; do paths+=(-e "$path"); done
    conda_cmd run -p "$env_prefix" python -m pip install --no-deps "${paths[@]}"

    if [[ "$profile" == linux-cuda ]]; then
        if [[ "$project" == rfdetr || "$project" == ultralytics ||
              "$project" == yolox ]]; then
            validate_linux_gpu "$env_prefix" "$project"
            if [[ "$project" == rfdetr ]]; then
                conda_cmd run -p "$env_prefix" python \
                    "$workspace_root/scripts/check_renku_rfdetr.py"
            fi
        fi
        conda_cmd run -p "$env_prefix" python -m pip check
    elif [[ "$project" != yolox ]]; then
        conda_cmd run -p "$env_prefix" python -m pip check
    fi

    if [[ "$project" != detection_common ]]; then
        local kernel="object-detection-$project"
        local display="Object Detection $project [$workspace_id] ($profile)"
        conda_cmd run -p "$env_prefix" python -m ipykernel install \
            --prefix "$env_prefix" --name "$kernel" --display-name "$display" \
            --env PYTHONPATH "" --env PYTHONNOUSERSITE "1"
    fi
    printf '%s\n' "$manifest_hash" > "$prefix/.object-detection-manifest"
    du -sh "$prefix"
}

if [[ $# -eq 0 || "$1" == all ]]; then
    set -- "${subprojects[@]}"
fi
for subproject in "$@"; do
	setup_one "$subproject";
done

# Set a custom conda env as the default, to prevent `base` being contaminated.
default_conda_env="object-detection-notebooks"
conda_cmd config --set default_activation_env ${default_conda_env}
