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

# Create or refresh one Conda environment from its platform manifest.
ensure_environment() {
    local env_prefix="$1"
    local definition="$2"
    local manifest_hash="$3"

    if conda_cmd run -p "$env_prefix" python -c 'pass' >/dev/null 2>&1; then
        if [[ ! -f "$env_prefix/.object-detection-manifest" ]] ||
            [[ "$(cat "$env_prefix/.object-detection-manifest")" != \
                "$manifest_hash" ]]; then
            conda_cmd env update -p "$env_prefix" -f "$definition"
        fi
    else
        conda_cmd env create -p "$env_prefix" -f "$definition"
    fi
}

# Run any installation step owned by one project.
install_project_hook() {
    local project="$1"
    local env_prefix="$2"

    case "$project" in
        yolox)
            bash "$workspace_root/scripts/setup/install_yolox.sh" "$env_prefix"
            ;;
    esac
}

# Install first-party packages required by one project.
install_workspace_packages() {
    local project="$1"
    local env_prefix="$2"
    if [[ "$project" == notebook-tools ]]; then
        return
    fi

    local installs=("$workspace_root/detection_common")
    case "$project" in
        detection_common)
            ;;
        evaluation)
            installs+=("$workspace_root/evaluation")
            ;;
        dataset)
            installs+=("$workspace_root/dataset")
            ;;
        rfdetr)
            installs+=(
                "$workspace_root/evaluation"
                "$workspace_root/dataset"
                "$workspace_root/rfdetr"
            )
            ;;
        ultralytics)
            installs+=(
                "$workspace_root/evaluation"
                "$workspace_root/dataset"
                "$workspace_root/ultralytics"
            )
            ;;
        yolox)
            installs+=(
                "$workspace_root/evaluation"
                "$workspace_root/dataset"
                "$workspace_root/yolox"
            )
            ;;
    esac

    local paths=()
    local path
    for path in "${installs[@]}"; do
        paths+=(-e "$path")
    done
    conda_cmd run -p "$env_prefix" python -m pip install --no-deps "${paths[@]}"
}

# Validate packages and platform behavior in one environment.
validate_environment() {
    local project="$1"
    local env_prefix="$2"

    if [[ "$profile" == linux-cuda &&
          "$project" =~ ^(rfdetr|ultralytics|yolox)$ ]]; then
        conda_cmd run -p "$env_prefix" python \
            "$workspace_root/scripts/check_linux_gpu.py" "$project"
        if [[ "$project" == rfdetr ]]; then
            conda_cmd run -p "$env_prefix" python \
                "$workspace_root/scripts/check_renku_rfdetr.py"
        fi
    fi
    conda_cmd run -p "$env_prefix" python -m pip check
}

# Register a Jupyter kernel for projects that own notebooks.
register_kernel() {
    local project="$1"
    local env_prefix="$2"

    if [[ "$project" == detection_common ||
          "$project" == notebook-tools ]]; then
        return
    fi

    local kernel="object-detection-$project"
    local display="Object Detection $project [$workspace_id] ($profile)"
    conda_cmd run -p "$env_prefix" python -m ipykernel install \
        --prefix "$env_prefix" --name "$kernel" --display-name "$display" \
        --env PYTHONPATH "" --env PYTHONNOUSERSITE "1"
}

# Run the complete setup lifecycle for one project.
setup_one() {
    local project="$1"
    local env_name
    env_name="$(environment_name "$project")" || { usage >&2; return 2; }
    local env_prefix="$workspace_root/.conda/envs/$env_name"
    local definition="$workspace_root/$project/environment-$profile.yml"
    [[ -f "$definition" ]] || {
        printf 'Missing manifest: %s\n' "$definition" >&2
        return 1
    }
    local manifest_hash
    manifest_hash="$(cksum "$definition" | awk '{print $1 ":" $2}')"

    register_envs_dir "${env_prefix%/*}"
    ensure_environment "$env_prefix" "$definition" "$manifest_hash"
    install_project_hook "$project" "$env_prefix"
    install_workspace_packages "$project" "$env_prefix"
    validate_environment "$project" "$env_prefix"
    register_kernel "$project" "$env_prefix"
    printf '%s\n' "$manifest_hash" > "$env_prefix/.object-detection-manifest"
    du -sh "$env_prefix"
}

if [[ $# -eq 0 || "$1" == all ]]; then
    set -- "${subprojects[@]}"
fi
for subproject in "$@"; do
    setup_one "$subproject"
done

# Set a custom conda env as the default, to prevent `base` being contaminated.
default_conda_env="object-detection-notebooks"
conda_cmd config --set default_activation_env "$default_conda_env"
