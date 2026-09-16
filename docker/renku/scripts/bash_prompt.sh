# Match the interactive prompt used by ~/.zsh_prompt_rc.
[[ $- == *i* ]] || return

__object_detection_update_prompt() {
    local previous_status=$?
    local directory_colour='\[\e[34m\]'
    local conda_colour='\[\e[32m\]'
    local git_colour='\[\e[35m\]'
    local prompt_colour='\[\e[36m\]'
    local reset='\[\e[0m\]'
    local separator=' '
    local conda_part=''
    local git_part=''
    local prompt_parts=''
    local git_ref=''

    if [[ -n ${CONDA_DEFAULT_ENV:-} ]]; then
        conda_part="${conda_colour}◈ ${CONDA_DEFAULT_ENV}${reset}"
    fi

    git_ref="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)" \
        || git_ref="$(git rev-parse --quiet --short HEAD 2>/dev/null)" \
        || true
    if [[ -n ${git_ref} ]]; then
        git_part="${git_colour}⎇ ${git_ref}${reset}"
    fi

    prompt_parts="${conda_part}"
    if [[ -n ${git_part} ]]; then
        [[ -n ${prompt_parts} ]] && prompt_parts+="${separator}"
        prompt_parts+="${git_part}"
    fi
    [[ -n ${prompt_parts} ]] && prompt_parts+="${separator}"

    PS1="[${prompt_parts}${directory_colour}▸ \w${reset}${prompt_colour}]\n\\\$${reset} "
    return "${previous_status}"
}

# Keep prompt hooks installed by the base image.
case "$(declare -p PROMPT_COMMAND 2>/dev/null)" in
    'declare -a '*) PROMPT_COMMAND+=(__object_detection_update_prompt) ;;
    *) PROMPT_COMMAND="${PROMPT_COMMAND:+${PROMPT_COMMAND%;};}__object_detection_update_prompt" ;;
esac
