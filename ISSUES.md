## DeepSeek-OCR — Known Issues and Fixes

This document collects install and runtime issues you may encounter (uv-based setup), with concrete fixes/workarounds.

### 1) vLLM upgrades torch to CUDA 12 runtime
**Symptom**
- After installing vLLM nightly, `torch`/`torchvision`/`torchaudio` are upgraded to CUDA 12.x runtime wheels (e.g., torch==2.9.0), even if you first installed CUDA 11.8 wheels.

**Cause**
- vLLM nightly depends on CUDA 12 runtime wheels and pulls a compatible PyTorch.

**Fix/Workarounds**
- Accept the upgrade if you plan to use vLLM. Ensure your NVIDIA driver supports CUDA 12 runtime.
- If you must keep `torch==2.6.0+cu118` for other reasons, use two environments: one for vLLM, one for Transformers.

### 2) CUDA driver mismatch or initialization failure
**Symptom**
- Errors like “CUDA initialization error”, “no CUDA GPUs are available”, or driver/runtime mismatch after vLLM (CUDA 12) install.

**Cause**
- The installed CUDA runtime (12.x) requires a sufficiently new NVIDIA driver.

**Fix**
- Update GPU drivers to a version compatible with CUDA 12.x.
- Alternatively, avoid vLLM nightly and stick to CUDA 11.8 PyTorch-only workflow.

### 3) flash-attn build takes long or fails
**Symptom**
- `uv pip install flash-attn==2.7.3 --no-build-isolation` takes a long time to build or fails with compiler/CUDA errors.

**Cause**
- flash-attn often builds native CUDA extensions from source.

**Fix/Workarounds**
- Ensure enough RAM and disk space; building can take >10 minutes.
- If build fails, upgrade `pip`, `setuptools`, `wheel`, and verify you are inside the intended venv:
  ```bash
  uv pip install --upgrade pip setuptools wheel
  which python  # should be .venv/bin/python
  ```
- If you continue to see build errors, consider pinning torch to a version known to work with flash-attn on your platform, or skip flash-attn for vLLM-only usage.

### 4) Transformers and Tokenizers version drift
**Symptom**
- Import errors or subtle runtime issues after installing vLLM (which may install newer `transformers`) when the project expects:
  - `transformers==4.46.3`
  - `tokenizers==0.20.3`

**Fix**
- Reinstall the project requirements after vLLM:
  ```bash
  uv pip install -r requirements.txt
  ```
- If runtime still complains about versions, create a dedicated env for Transformers usage with the exact pins from `requirements.txt`.

### 5) PyTorch wheel index not found
**Symptom**
- Installing PyTorch without specifying the index URL fails or pulls CPU-only wheels unintentionally.

**Fix**
- Always specify the CUDA wheel index when pinning CUDA 11.8:
  ```bash
  uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118
  ```

### 6) ImportError: xformers/flash-attn/flashinfer not found
**Symptom**
- vLLM may try to use optional acceleration backends; import fails at runtime.

**Fix**
- These are optional. If import fails, vLLM falls back; performance may be lower. To enable specific backends, install the relevant wheels supported by your CUDA/driver stack (the nightly vLLM wheels often bring many dependencies already).

### 7) OOM or high VRAM usage
**Symptom**
- Out-of-memory errors on limited GPUs.

**Fixes/Workarounds**
- Reduce batch size, image size, and max tokens.
- For vLLM, tune `--max-model-len`, use fewer concurrent requests.
- For Transformers, lower `base_size`, `image_size`, disable extra processing.

### 8) PDF/image dependencies
**Symptom**
- Errors parsing PDFs or images.

**Fix**
- Ensure project requirements are installed:
  ```bash
  uv pip install -r requirements.txt
  ```
- If system-level image/PDF dependencies are missing on very minimal OS images, install standard Linux packages for image codecs and fonts. Typically not needed on mainstream distros since Python wheels (e.g., `pymupdf`, `pikepdf`, `Pillow`) bundle what’s needed.

### 9) “trust_remote_code” behavior with Transformers
**Symptom**
- Model load warnings or failures without `trust_remote_code=True`.

**Fix**
- Follow README examples and set `trust_remote_code=True` when loading the model with Transformers.

### 10) General troubleshooting checklist
- Confirm you’re in the correct venv:
  ```bash
  which python
  python -V
  ```
- Print versions:
  ```bash
  python - <<'PY'
import torch, transformers, tokenizers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
try:
    import vllm; print("vllm:", vllm.__version__)
except Exception as e:
    print("vllm import issue:", e)
try:
    import flash_attn; print("flash-attn: OK")
except Exception as e:
    print("flash-attn issue:", e)
PY
  ```
- If versions drift, reinstall `requirements.txt` and/or split into separate envs for vLLM vs Transformers.

### 11) Transformers path: missing matplotlib
**Symptom**
- ImportError during model load from hub: “This modeling file requires ... matplotlib.”

**Cause**
- The model’s dynamic modules reference matplotlib for visualization utilities.

**Fix**
```bash
uv pip install matplotlib
```

### 12) Transformers path: Accelerate required for device_map
**Symptom**
- ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>=0.26.0'`

**Cause**
- Passing `device_map` (or implicit low memory loading) requires the Accelerate library.

**Fix**
```bash
uv pip install 'accelerate>=0.26.0'
```

### 13) Transformers path: Flash Attention warnings (dtype/device)
**Symptom**
- Warnings about using FA2 without specifying dtype or not initializing model on GPU.

**Cause**
- Model not explicitly moved to GPU, or dtype not set for Flash Attention.

**Fix/Workaround**
- Initialize with explicit dtype and device mapping:
  - In scripts we set: `torch_dtype=torch.bfloat16, device_map='cuda'` and `model = model.eval().to('cuda')`.

### 14) CUDA OOM during Transformers load/inference
**Symptom**
- `torch.OutOfMemoryError` while loading weights or running inference.

**Causes**
- Insufficient free VRAM; multiple processes occupying GPU; fragmentation.

**Fix/Workarounds**
- Free up GPU memory (kill other processes).
- Set environment to reduce fragmentation:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
- Lower workload:
  - Use smaller `base_size`/`image_size` and enable `crop_mode` in scripts (defaults provided in `scripts/extract_pdf.sh`).
- Use the vLLM pipeline if feasible (often better memory behavior), in a separate env if version constraints apply.

### 15) vLLM path: ImportError DeepseekV3Config (Transformers compatibility)
**Symptom**
- `ImportError: cannot import name 'DeepseekV3Config' from 'transformers'` when launching vLLM scripts.

**Cause**
- Installed `transformers` version too old for current vLLM nightly expectations.

**Fix/Workarounds**
- Upgrade `transformers` (e.g., `uv pip install -U transformers`), BUT this can conflict with repo pins used by the Transformers path.
- Recommended: create a separate venv for vLLM nightly usage to allow `transformers` upgrades without impacting the Transformers path:
  - Env A (Transformers path): `transformers==4.46.3`, `tokenizers==0.20.3`
  - Env B (vLLM nightly): accept `transformers` version required by vLLM and CUDA 12 torch wheels

### 16) “Model type deepseek_vl_v2 to instantiate DeepseekOCR” warning
**Symptom**
- Warning printed when loading model from hub stating mismatch of model types.

**Cause**
- The hub repo uses custom modeling code (`trust_remote_code=True`) that maps configs to a custom class.

**Fix/Workaround**
- This is a warning; proceed. If it escalates to an error after a `transformers` upgrade, pin to a previously working revision or split envs per Section 15.

### 17) Suppressing advisory warnings from Transformers
**Symptom**
- Repeated advisory warnings like model type mismatch clutter the logs.

**Fix/Workaround**
- Set environment variable and lower verbosity:
  ```bash
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  ```
  In scripts, set:
  ```python
  from transformers.utils import logging as hf_logging
  hf_logging.set_verbosity_error()
  ```
  The provided wrappers (`scripts/extract_pdf.sh`, `scripts/ocr_pdf_hf.py`) already apply these.


