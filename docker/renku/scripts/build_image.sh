#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 IMAGE [--load|--push]" >&2
}

if (( $# < 1 || $# > 2 )); then
    usage
    exit 2
fi

readonly image="$1"
readonly output="${2:---load}"

case "${output}" in
    --load | --push) ;;
    *)
        usage
        exit 2
        ;;
esac

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly context_dir="$(cd -- "${script_dir}/.." && pwd)"

exec docker buildx build \
    --platform linux/amd64 \
    --tag "${image}" \
    "${output}" \
    "${context_dir}"
