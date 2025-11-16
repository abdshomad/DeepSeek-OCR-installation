#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/run_ocr_pdf_hf.sh <input_pdf> <output_dir> [base_size] [image_size] [crop_mode]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

uv run python scripts/ocr_pdf_hf.py "$@"


