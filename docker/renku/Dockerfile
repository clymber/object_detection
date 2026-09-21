FROM renku/renkulab-vscode:ubuntu-9c147f6

# Use root user to install system dependencies and copy scripts
USER root
COPY --chmod=0755 scripts/start_tunnel.sh /usr/local/bin/start_tunnel
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git jq tmux vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the vscode user for subsequent operations
USER vscode

# Install the VS Code CLI and Tunnel extensions for the vscode user.
ENV CODE_CLI_PATH="/home/vscode/.local/bin/code" \
    VSCODE_EXTENSIONS_DIR="/home/vscode/.vscode_tunnel/extensions"
RUN --mount=type=bind,source=./scripts/install_vscode_extensions.sh,target=/tmp/install_vscode_extensions.sh \
    bash /tmp/install_vscode_extensions.sh

# Install Miniforge for the vscode user
RUN --mount=type=bind,source=./scripts/install_miniforge.sh,target=/tmp/install_miniforge.sh \
    bash /tmp/install_miniforge.sh
ENV PATH="/home/vscode/.local/bin:/home/vscode/miniforge3/bin:${PATH}" \
    CONDA_BIN="/home/vscode/miniforge3/bin/conda"

# Configure Conda, the Bash prompt, and optional Roboflow credentials.
COPY --chown=vscode:vscode --chmod=0644 scripts/bash_prompt.sh /home/vscode/.bash_prompt_rc
RUN --mount=type=bind,source=./scripts/configure_bashrc.sh,target=/tmp/configure_bashrc.sh \
    bash /tmp/configure_bashrc.sh

# Install Codex for the vscode user.
RUN curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/install-codex.sh \
    && sh /tmp/install-codex.sh \
    && rm /tmp/install-codex.sh \
    && codex --version
