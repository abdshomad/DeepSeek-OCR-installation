import os
import io
import sys
from typing import List

import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

# Suppress advisory warnings such as model type mismatch messages
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
hf_logging.set_verbosity_error()


def pdf_to_images(pdf_path: str, dpi: int = 144) -> List[Image.Image]:
    images: List[Image.Image] = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)
    pdf_document.close()
    return images


def main():
    if len(sys.argv) < 3:
        print("Usage: python ocr_pdf_hf.py <input_pdf> <output_dir> [base_size] [image_size] [crop_mode]")
        sys.exit(1)

    input_pdf = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    base_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    image_size = int(sys.argv[4]) if len(sys.argv) > 4 else 640
    crop_mode = (sys.argv[5].lower() == "true") if len(sys.argv) > 5 else True

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    model_name = "deepseek-ai/DeepSeek-OCR"
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = model.eval().to("cuda")

    print(f"Reading PDF: {input_pdf}")
    images = pdf_to_images(input_pdf, dpi=144)

    combined_mmd_path = os.path.join(
        output_dir, os.path.basename(input_pdf).replace(".pdf", ".mmd")
    )

    contents: List[str] = []

    for idx, image in enumerate(images):
        print(f"Processing page {idx+1}/{len(images)} ...")
        # Save the page image for reference
        page_img_path = os.path.join(images_dir, f"page_{idx+1}.jpg")
        image.save(page_img_path, format="JPEG", quality=95)

        # Run inference
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=page_img_path,
            output_path=output_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
            test_compress=True,
        )

        # res may be written to files already by the model; we append a page split marker
        contents.append(str(res) if res is not None else "")
        contents.append("\n<--- Page Split --->\n")

    with open(combined_mmd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(contents))

    print(f"Done. Combined markdown: {combined_mmd_path}")


if __name__ == "__main__":
    main()


