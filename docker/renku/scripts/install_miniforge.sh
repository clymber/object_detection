#!/usr/bin/env bash
# Install Miniforge for current user.
set -euo pipefail

readonly username="vscode" # Inherited from image renku/renkulab-vscode
readonly version="26.7.2-0"

readonly prefix="/home/${username}/miniforge3"
readonly installer="Miniforge3-${version}-Linux-x86_64.sh"
readonly release="https://github.com/conda-forge/miniforge/releases/download/${version}"

# Prevent reinstallation
if [[ -e "${prefix}" ]]; then
    if [[ -x "${prefix}/bin/conda" ]]; then
        echo "Miniforge is already installed: $(${prefix}/bin/conda --version)"
        exit 0
    fi
fi

workdir="$(mktemp -d /tmp/install_miniforge.XXXXXX)"
trap '[[ -d "${workdir}" ]] && rm -rf -- "${workdir}"' EXIT

curl -fsSL --retry 3 -o "$workdir/$installer" "$release/$installer"
curl -fsSL --retry 3 -o "$workdir/$installer.sha256" "$release/$installer.sha256"

(
    cd "$workdir" && sha256sum -c "$installer.sha256"
)

bash "$workdir/$installer" -b -p "$prefix"
"$prefix/bin/conda" --version
printf 'Miniforge installed at %s\n' "$prefix"
