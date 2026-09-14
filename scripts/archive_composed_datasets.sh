#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
data_root="${OBJECT_DETECTION_DATA_ROOT:-$workspace_root/data}"
if [[ "$data_root" != /* ]]; then
    printf 'OBJECT_DETECTION_DATA_ROOT must be an absolute path.\n' >&2
    exit 2
fi
if [[ ! -d "$data_root/composed" ]]; then
    printf 'Composed datasets not found: %s\n' "$data_root/composed" >&2
    exit 1
fi

COPYFILE_DISABLE=1 tar \
    --no-xattrs \
    --exclude='*.cache' \
    --exclude='._*' \
    --exclude='.DS_Store' \
    -czf "$data_root/composed_datasets.tgz" \
    -C "$data_root" \
    composed
