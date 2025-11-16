#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/extract_pdf.sh /absolute/path/to/file.pdf [base_size] [image_size] [crop_mode]
# Output:
#   Writes <basename>.mmd next to the input PDF path and any page images to <dir>/images.

if [ $# -lt 1 ]; then
  echo "Usage: scripts/extract_pdf.sh <input_pdf> [base_size] [image_size] [crop_mode]" >&2
  exit 1
fi

INPUT_PDF="$(realpath "$1")"
BASE_SIZE="${2:-1024}"
IMAGE_SIZE="${3:-640}"
CROP_MODE="${4:-true}"

if [ ! -f "$INPUT_PDF" ]; then
  echo "Input PDF not found: $INPUT_PDF" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="$(dirname "$INPUT_PDF")"
BASENAME="$(basename "$INPUT_PDF")"
MMD_PATH="${OUTPUT_DIR}/${BASENAME%.pdf}.mmd"

echo "Extracting OCR markdown for: $INPUT_PDF"
echo "Output directory: $OUTPUT_DIR"

# Improve CUDA allocator behavior and silence Transformers advisory warnings
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

uv run python scripts/ocr_pdf_hf.py "$INPUT_PDF" "$OUTPUT_DIR" "$BASE_SIZE" "$IMAGE_SIZE" "$CROP_MODE"

if [ -f "$MMD_PATH" ]; then
  echo "Markdown written to: $MMD_PATH"
else
  echo "Expected markdown not found at: $MMD_PATH" >&2
  exit 2
fi


