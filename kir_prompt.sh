#!/usr/bin/env bash
#
# kir_prompt.sh — Oxford blue & gold bash prompt with git branch awareness
#
# Usage:
#   Add this line to your ~/.bashrc:
#     source /path/to/kir_prompt.sh
#
# Produces a prompt of the form:
#   user@hostname [git-branch] path $
#
# Colour scheme:
#   Oxford Blue  (#002147 -> approximated as 256-colour 17/18/24)
#   Oxford Gold  (#A39161 -> approximated as 256-colour 178/179)

# --- Only run in interactive bash shells ---
[[ $- != *i* ]] && return

# Stop venv/virtualenv's activate script from prepending its own (env) prefix —
# we render it ourselves below, styled to match the colour scheme.
export VIRTUAL_ENV_DISABLE_PROMPT=1

# --- 256-colour escape codes ---
# Oxford Blue (deep navy)

KIR_BLUE='\[\e[38;5;68m\]'
_KIR_BLUE_BOLD='\[\e[1;38;5;68m\]'

# Oxford Gold (muted gold/ochre)
_KIR_GOLD='\[\e[38;5;178m\]'
_KIR_GOLD_BOLD='\[\e[1;38;5;178m\]'
# Supporting colours
_KIR_GREY='\[\e[38;5;245m\]'
_KIR_RED='\[\e[38;5;160m\]'
_KIR_GREEN='\[\e[38;5;108m\]'
_KIR_RESET='\[\e[0m\]'

# --- Python virtual environment helper ---
# Prints "(envname) " in green if a venv/virtualenv is active.
# Works with venv, virtualenv, and conda (CONDA_DEFAULT_ENV) — falls back
# to the directory name if VIRTUAL_ENV_PROMPT isn't set.
__kir_venv() {
    local env_name=""

    if [[ -n "$VIRTUAL_ENV" ]]; then
        env_name="${VIRTUAL_ENV_PROMPT:-$(basename "$VIRTUAL_ENV")}"
    elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        env_name="$CONDA_DEFAULT_ENV"
    fi

    [[ -n "$env_name" ]] && printf '%s(%s) %s' "${_KIR_GREEN}" "${env_name}" "${_KIR_RESET}"
}

# --- Git branch / status helper ---
# Prints "[branch]" in gold, or "[branch*]" in red if there are uncommitted changes.
# Silent (prints nothing) outside a git repo.
__kir_git_branch() {
    local branch
    branch=$(git symbolic-ref --short HEAD 2>/dev/null) || \
        branch=$(git rev-parse --short HEAD 2>/dev/null) || return

    local dirty=""
    if ! git diff --quiet --ignore-submodules HEAD 2>/dev/null; then
        dirty="*"
    elif [[ -n $(git status --porcelain 2>/dev/null) ]]; then
        dirty="*"
    fi

    if [[ -n "$dirty" ]]; then
        printf ' %s[%s%s]%s' "${_KIR_RED}" "${branch}" "${dirty}" "${_KIR_RESET}"
    else
        printf ' %s[%s]%s' "${_KIR_GOLD}" "${branch}" "${_KIR_RESET}"
    fi
}

# --- Build the prompt ---
# Format: user@hostname [branch] ~/path $
__kir_set_prompt() {
    local exit_code=$?

    PS1=""
    PS1+="$(__kir_venv)"                          # (venv) prefix, if active
    PS1+="${_KIR_BLUE_BOLD}\u${_KIR_RESET}"      # username in Oxford blue (bold)
    PS1+="${_KIR_GREY}@${_KIR_RESET}"
    PS1+="${_KIR_BLUE_BOLD}\h${_KIR_RESET}"      # hostname in Oxford blue (bold)
    PS1+="$(__kir_git_branch)"                   # git branch, if applicable
    PS1+=" ${_KIR_GOLD}\w${_KIR_RESET}"          # current path in gold
    if [[ $exit_code -ne 0 ]]; then
        PS1+=" ${_KIR_RED}✗${_KIR_RESET}"        # mark non-zero exit status
    fi
    PS1+="\n${_KIR_BLUE}\$${_KIR_RESET} "        # prompt symbol on its own line
}

PROMPT_COMMAND=__kir_set_prompt

# Additional Paths to $PATH
export PATH=/lus/lfs1aip2/projects/u6pl/software/bin:$PATH
export PATH=/lus/lfs1aip2/projects/u6pl/software/nvim-linux-arm64/bin:$PATH

# Mac terminals have trouble with tmux backspacing and it moves the cursor forward. Following should fix it
export TERMINFO_DIRS="/lus/lfs1aip2/projects/u6pl/software/tmux-3.7b/share/terminfo:"
# Aliases
alias ls='ls --color=auto'

# Slurm variables
export SACCT_FORMAT="jobid%-14,jobname%-15,user%-9,start%12,elapsed,avecpu,mincpu,totalcpu,alloccpus%5,ntasks%5,reqmem%7,maxrss,state%-10,nodelist%-30"
export SQUEUE_FORMAT="%13i %8u %9a %12j%.4C %.7m %7P %12S %.11L %8T %20R"

# Bind Apptainer
export APPTAINER_BIND="/lus"
