#!/usr/bin/env bash
# Run nb01 through nb04 in notebook-number order, retaining failed sessions.
set -euo pipefail

if [[ $# -gt 0 ]]; then
    printf 'Usage: %s\n' "$0"
    printf 'Runs project nb01* through nb04*.ipynb notebooks sequentially.\n'
    [[ $# -eq 1 && ( "$1" == -h || "$1" == --help ) ]] && exit 0
    exit 2
fi

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$workspace_root/scripts/run_notebook_tmux.sh"
cd "$workspace_root"
command -v tmux >/dev/null || { printf 'tmux is required.\n' >&2; exit 1; }

shopt -s nullglob
notebooks=(
    {dataset,evaluation,rfdetr,ultralytics,yolox}/notebooks/nb0[1-5]*.ipynb
)
if [[ ${#notebooks[@]} -eq 0 ]]; then
    printf 'No nb01* through nb05*.ipynb notebooks found.\n' >&2
    printf 'Generate the notebooks first with make sync-notebooks.\n' >&2
    exit 1
fi
# All paths have three components; sort by notebook name across projects.
mapfile -d '' -t notebooks < <(
    printf '%s\0' "${notebooks[@]}" | LC_ALL=C sort -z -t / -k3,3V
)

batch="$(date -u +%Y%m%dT%H%M%S)-$$"
result=0
index=0
for notebook in "${notebooks[@]}"; do
    index=$((index + 1))
    name="${notebook##*/}"
    name="${name%.ipynb}"
    session="${name//[^[:alnum:]_-]/-}-$batch-$index"
    printf '\n[%s/%s] Running %s\n' "$index" "${#notebooks[@]}" "$notebook"
    if ! bash "$runner" run --file "$notebook" --session "$session"; then
        printf 'Error starting %s; continuing to the next notebook.\n' \
            "$notebook" >&2
        result=1
        continue
    fi

    while true; do
        sleep 10
        # Read completion first so a notebook finishing during the check
        # cannot be mistaken for success before its exit status is checked.
        if ! dead="$(tmux display-message -p -t "=$session:" '#{pane_dead}')"; then
            printf 'Error inspecting %s; continuing to the next notebook.\n' \
                "$session" >&2
            result=1
            break
        fi
        if bash "$runner" check --session "$session"; then
            if [[ "$dead" == 1 ]]; then
                if tmux kill-session -t "=$session"; then
                    printf 'Completed %s; removed session %s.\n' \
                        "$notebook" "$session"
                else
                    printf 'Error removing completed session %s.\n' "$session" >&2
                    result=1
                fi
                break
            fi
        else
            status=$?
            printf 'Error: %s exited or check failed (status %s).\n' \
                "$notebook" "$status" >&2
            printf 'Keeping session %s; continuing to the next notebook.\n' \
                "$session" >&2
            result=1
            break
        fi
    done
done

printf '\nFinished processing %s notebooks.\n' "${#notebooks[@]}"
exit "$result"
