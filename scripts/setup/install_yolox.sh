#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s ENV_PREFIX\n' "$0" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly workspace_root="$(cd "$script_dir/../.." && pwd)"
readonly env_prefix="$1"
readonly yolox_relative_dir="thirdparty/YOLOX"
readonly yolox_source_dir="$workspace_root/$yolox_relative_dir"
yolox_build_dir=""

# Remove only the temporary directory created by this installer.
cleanup() {
    if [[ "$yolox_build_dir" == /tmp/object-detection-yolox.* &&
          -d "$yolox_build_dir" ]]; then
        rm -rf -- "$yolox_build_dir"
    fi
}
trap cleanup EXIT

# Run the target environment's Python without inherited virtual-environment state.
env_python() {
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u PIP_PREFIX \
        -u PIP_TARGET -u PIP_USER "$env_prefix/bin/python" "$@"
}

# Initialize the pinned YOLOX submodule when needed and print its commit.
resolve_yolox_commit() {
    command -v git >/dev/null || {
        printf 'git is required to initialize and build YOLOX.\n' >&2
        return 1
    }
    if [[ ! -f "$yolox_source_dir/setup.py" ]]; then
        git -C "$workspace_root" submodule update --init --depth 1 -- \
            "$yolox_relative_dir" >&2
    fi
    [[ -f "$yolox_source_dir/setup.py" ]] || {
        printf 'YOLOX submodule is unavailable at %s.\n' "$yolox_source_dir" >&2
        return 1
    }

    local expected_commit
    expected_commit="$(
        git -C "$workspace_root" ls-files --stage -- "$yolox_relative_dir" |
            awk '$1 == "160000" {print $2}'
    )"
    local source_commit
    source_commit="$(git -C "$yolox_source_dir" rev-parse HEAD)"
    [[ -n "$expected_commit" && "$source_commit" == "$expected_commit" ]] || {
        printf 'YOLOX submodule is not at the pinned workspace revision.\n' >&2
        printf 'Run: git submodule update --init -- %s\n' \
            "$yolox_relative_dir" >&2
        return 1
    }
    printf '%s\n' "$source_commit"
}

# Install the pinned YOLOX source without modifying the submodule checkout.
install_yolox() {
    local commit
    commit="$(resolve_yolox_commit)"

    local check_commit
    check_commit=$(printf '%s\n' \
        "import yolox.layers.fast_cocoeval" \
        "from yolox._object_ctrl_build import SOURCE_COMMIT" \
        "assert SOURCE_COMMIT == '$commit'"
    )
    if env_python -c "$check_commit" >/dev/null 2>&1; then
        return 0
    fi

    command -v c++ >/dev/null || {
        printf 'A C++ compiler is required to install YOLOX.\n' >&2
        return 1
    }
    [[ -x "$env_prefix/bin/python" ]] || {
        printf 'Python is unavailable in YOLOX environment: %s\n' \
            "$env_prefix" >&2
        return 1
    }

    yolox_build_dir="$(mktemp -d /tmp/object-detection-yolox.XXXXXX)"
    git -C "$yolox_source_dir" archive "$commit" |
        tar -x -C "$yolox_build_dir"
    env_python - "$yolox_build_dir" "$commit" <<-'PY'
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
		excluded = {"opencv_python", "onnx-simplifier==0.4.10"}
		filtered = [line for line in requirements.splitlines() if line not in excluded]
		requirements_path.write_text("\n".join(filtered) + "\n")
		(checkout / "yolox/_object_ctrl_build.py").write_text(
		    f'SOURCE_COMMIT = "{commit}"\nCXX_STANDARD = "c++17"\n'
		)
	PY

    env_python -m pip install --no-build-isolation --no-deps \
        --force-reinstall "$yolox_build_dir"
    env_python -c "$check_commit"
}

install_yolox
