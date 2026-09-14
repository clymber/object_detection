#!/usr/bin/env bash
# Run a project-owned notebook through its Conda kernel in a detached session.
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
    printf 'Usage: %s run --file <project>/notebooks/<file>.ipynb [--session NAME]\n' "$0"
    printf '       %s check --session NAME\n' "$0"
}
[[ $# -gt 0 ]] || { usage >&2; exit 2; }
action="$1"
shift
[[ "$action" == -h || "$action" == --help ]] && { usage; exit 0; }
notebook=""
session=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; notebook="$2"; shift 2 ;;
        -s|--session) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; session="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
command -v tmux >/dev/null || { printf 'tmux is required.\n' >&2; exit 1; }

case "$action" in
    run)
        case "$notebook" in
            dataset/notebooks/*.ipynb|evaluation/notebooks/*.ipynb|rfdetr/notebooks/*.ipynb|ultralytics/notebooks/*.ipynb|yolox/notebooks/*.ipynb) ;;
            *) printf 'Use a project-owned .ipynb path relative to the workspace root.\n' >&2; exit 2 ;;
        esac
        [[ -f "$workspace_root/$notebook" ]] || { printf 'Notebook not found: %s\n' "$notebook" >&2; exit 1; }
        if [[ -z "$session" ]]; then
            session="$(basename "$notebook" .ipynb)"
            session="${session//[^[:alnum:]_-]/-}"
        fi
        ;;
    check) [[ -n "$session" ]] || { usage >&2; exit 2; } ;;
    *) usage >&2; exit 2 ;;
esac
[[ "$session" =~ ^[[:alnum:]_-]+$ ]] || { printf 'Invalid session name.\n' >&2; exit 2; }
log="$workspace_root/outputs/notebook_logs/$session.log"

if [[ "$action" == check ]]; then
    tmux has-session -t "=$session" 2>/dev/null || { printf 'Session not found: %s\n' "$session" >&2; exit 1; }
    pane="$(tmux display-message -p -t "=$session:" '#{pane_dead} #{pane_dead_status}')"
    read -r dead status <<< "$pane"
    if [[ "$dead" == 0 ]]; then
        printf 'Session %s is running. Log: %s\n' "$session" "$log"
        exit 0
    fi
    if [[ ! "$status" =~ ^[0-9]+$ ]]; then
        last_line="$(tail -n 1 "$log" 2>/dev/null || true)"
        if [[ "$last_line" == __OBJECT_DETECTION_EXIT_STATUS__=* ]]; then
            status="${last_line#*=}"
        else
            status=1
        fi
    fi
    printf 'Session %s exited with status %s. Log: %s\n' "$session" "$status" "$log"
    exit "$status"
fi

tmux has-session -t "=$session" 2>/dev/null && {
    printf 'Session already exists: %s\n' "$session" >&2
    exit 1
}
mkdir -p "$(dirname "$log")"
environment='unset VIRTUAL_ENV PYTHONPATH PYTHONHOME PIP_PREFIX PIP_TARGET PIP_USER; export PYTHONNOUSERSITE=1; '
for key in OBJECT_DETECTION_WORKSPACE_ROOT OBJECT_DETECTION_DATA_ROOT \
    OBJECT_DETECTION_OUTPUT_ROOT CUDA_VISIBLE_DEVICES NOTEBOOK_TIMEOUT \
    RF_HOME HF_HOME TORCH_HOME XDG_CACHE_HOME MPLCONFIGDIR MPLBACKEND \
    ${!RFDETR_@} ${!YOLOX_NANO_@} ${!YOLOX_TINY_@} ${!ULTRALYTICS_@}; do
    if [[ ${!key+x} ]]; then
        printf -v assignment 'export %s=%q; ' "$key" "${!key}"
        environment+="$assignment"
    fi
done
printf -v runner 'bash %q %q' "$workspace_root/scripts/run_local_notebook.sh" "$notebook"
printf -v notebook_command '%s%s 2>&1 | tee %q; status=$?; echo __OBJECT_DETECTION_EXIT_STATUS__=$status >> %q; exit $status' "$environment" "$runner" "$log" "$log"
printf -v tmux_command 'bash -o pipefail -c %q' "$notebook_command"
tmux new-session -d -s "$session" -c "$workspace_root" "$tmux_command" \; \
    set-option -t "=$session:" remain-on-exit on
printf 'Started %s. Log: %s\n' "$session" "$log"
printf 'Check with: %s check --session %s\n' "$0" "$session"
