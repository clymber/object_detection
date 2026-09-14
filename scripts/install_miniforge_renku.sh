#!/usr/bin/env bash
# Install a pinned Miniforge release on Renku's persistent work volume.
set -euo pipefail

readonly prefix="/home/renku/work/miniforge3"
readonly version="26.7.2-0"
readonly installer="Miniforge3-${version}-Linux-x86_64.sh"
readonly release="https://github.com/conda-forge/miniforge/releases/download/${version}"

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    printf 'This installer requires Linux x86_64.\n' >&2
    exit 2
}
if [[ -e "$prefix" ]]; then
    [[ -x "$prefix/bin/conda" ]] || {
        printf 'Refusing to replace an unrecognized directory: %s\n' "$prefix" >&2
        exit 1
    }
    "$prefix/bin/conda" --version
    exit 0
fi

command -v curl >/dev/null || { printf 'curl is required.\n' >&2; exit 1; }
command -v sha256sum >/dev/null || { printf 'sha256sum is required.\n' >&2; exit 1; }
work_dir="$(mktemp -d /tmp/object-detection-miniforge.XXXXXX)"
trap 'rm -rf -- "$work_dir"' EXIT

curl -fsSL --retry 3 -o "$work_dir/$installer" "$release/$installer"
curl -fsSL --retry 3 -o "$work_dir/$installer.sha256" "$release/$installer.sha256"
(cd "$work_dir" && sha256sum -c "$installer.sha256")
bash "$work_dir/$installer" -b -p "$prefix"
"$prefix/bin/conda" --version
printf 'Miniforge installed at %s\n' "$prefix"
