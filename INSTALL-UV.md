## DeepSeek-OCR — Installation (uv-based, no conda)

This guide installs DeepSeek-OCR using uv virtual environments (no conda), matching the repository’s README while replacing conda with uv.

If you hit any problems, see ISSUES.md for known issues and fixes.

### 1) Prerequisites
- NVIDIA GPU with recent drivers
- Linux x86_64
- Internet access to fetch wheels

Note: The instructions use prebuilt wheels and do not require a system CUDA toolkit. CUDA runtime libraries are provided by installed Python wheels.

### 2) Install uv
If uv is not present:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version
```

### 3) Create a Python 3.12 environment
```bash
cd /home/aiserver/LABS/OCR/DEEPSEEK-OCR/DeepSeek-OCR
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
python -V
```

### 4) Install core packages
Two supported usage paths exist in this repo: vLLM inference and Transformers inference. To keep them compatible in one env, follow the sequence below.

1) Install PyTorch (CUDA-enabled wheels):
```bash
uv pip install --upgrade pip setuptools wheel
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

2) Install vLLM (upstream nightly as recommended by the README):
```bash
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```
Notes:
- vLLM may install newer CUDA 12 runtime wheels and upgrade torch. This is expected and acceptable for the vLLM path.

3) Install project requirements:
```bash
uv pip install -r requirements.txt
```
This ensures `transformers==4.46.3` and `tokenizers==0.20.3` match the repo.

4) Install flash-attn (for Transformers path):
```bash
uv pip install flash-attn==2.7.3 --no-build-isolation
```

### 5) Quick smoke test
Run the following to verify critical imports:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)

try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as e:
    print("vllm import issue:", e)

import transformers, tokenizers
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)

try:
    import flash_attn  # noqa: F401
    print("flash-attn: OK")
except Exception as e:
    print("flash-attn import issue:", e)

try:
    import accelerate  # noqa: F401
    import importlib
    acc_ver = importlib.metadata.version("accelerate")
    print("accelerate:", acc_ver)
except Exception as e:
    print("accelerate import issue:", e)
PY
```

### 6) Running

- vLLM examples (edit paths in `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`):
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
python run_dpsk_ocr_pdf.py
python run_dpsk_ocr_eval_batch.py
```

- Transformers examples:
```bash
cd /home/aiserver/LABS/OCR/DEEPSEEK-OCR/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

### Notes and compatibility
- If you must strictly keep `torch==2.6.0+cu118`, install vLLM in a separate environment, since nightly vLLM can upgrade torch to a newer CUDA 12 runtime wheel. Keeping both paths in a single env is possible but torch may be upgraded by vLLM.
- The repository’s README explicitly mentions you can ignore certain vLLM-versus-transformers version warnings when sharing the same environment.


