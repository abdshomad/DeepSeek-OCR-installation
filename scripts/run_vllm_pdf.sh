#!/usr/bin/env bash
set -euo pipefail

# Runs the vLLM-based PDF OCR using config in DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py
# Edit INPUT_PATH and OUTPUT_PATH in the config before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/DeepSeek-OCR-master/DeepSeek-OCR-vllm"

uv run python run_dpsk_ocr_pdf.py


