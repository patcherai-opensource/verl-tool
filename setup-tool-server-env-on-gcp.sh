#!/usr/bin/env bash
set -e  # Exit on error
set -x  # Print each command

# 0. Define constants
PYTHON_VERSION=3.11.12
VENV_DIR=tool-server

# 1. Install pyenv and dependencies
if ! command -v pyenv &>/dev/null; then
  curl https://pyenv.run | bash

  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"

  # Ensure dependencies are installed (Debian/Ubuntu)
  sudo apt update
  sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
fi

# 2. Install Python + virtualenv with pyenv
pyenv install -s "$PYTHON_VERSION"
pyenv virtualenv "$PYTHON_VERSION" "$VENV_DIR"
pyenv activate "$VENV_DIR"

# 3. Install poetry and plugin
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry self add poetry-plugin-export

# 4. Install uv
pip install uv

# 5. Install direnv
if ! command -v direnv &>/dev/null; then
  sudo apt install -y direnv
fi

# Add direnv hook to shell if not already present
SHELL_RC="$HOME/.bashrc"
if [[ "$SHELL" == *zsh ]]; then
  SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q 'direnv hook' "$SHELL_RC"; then
  echo 'eval "$(direnv hook bash)"' >> "$SHELL_RC"
fi
source "$SHELL_RC"

# 6. Install verl-tool in editable mode
uv pip install -e .

# 7. Install sister dependencies from vms
poetry --directory ../vms export --without-hashes --output /tmp/vms-requirements.txt
uv pip install -r /tmp/vms-requirements.txt

direnv allow

echo "✅ tool-server setup complete and environment activated."
