#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-OCR uv-based installer (no conda)
# Idempotent: can be re-run safely.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "[1/7] Checking/Installing uv..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

echo "[2/7] Creating Python 3.12 virtual environment..."
uv python install 3.12
if [ -d ".venv" ]; then
  # Ensure venv uses uv-managed 3.12
  # rm -rf .venv
fi
uv venv --python 3.12

echo "[3/7] Installing PyTorch 2.6.0 (CUDA 11.8 wheels)..."
uv pip install --upgrade pip setuptools wheel
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

echo "[4/7] Installing vLLM nightly (may upgrade torch to CUDA 12 runtime)..."
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

echo "[5/7] Installing project requirements..."
uv pip install -r requirements.txt

echo "[6/7] Installing flash-attn (Transformers path)..."
uv pip install flash-attn==2.7.3 --no-build-isolation

echo "[7/7] Running smoke test..."
uv run python - <<'PY'
import sys
ok = True
def pr(label, fn):
    global ok
    try:
        v = fn()
        print(f"{label}: {v}")
    except Exception as e:
        print(f"{label} import issue:", e)
        ok = False

def _torch():
    import torch
    return torch.__version__
def _vllm():
    import vllm
    return vllm.__version__
def _transformers():
    import transformers
    return transformers.__version__
def _tokenizers():
    import tokenizers
    return tokenizers.__version__
def _flash():
    import flash_attn
    return "OK"

pr("torch", _torch)
pr("vllm", _vllm)
pr("transformers", _transformers)
pr("tokenizers", _tokenizers)
pr("flash-attn", _flash)

sys.exit(0 if ok else 1)
PY

echo "Installation completed successfully."


