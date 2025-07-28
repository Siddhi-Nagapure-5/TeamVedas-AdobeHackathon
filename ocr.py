# ocr.py

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import time
import os
import json
from typing import List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

TESSERACT_CONFIG = "--oem 1 --psm 4"

OCRImageType = Tuple[int, Image.Image]
ExtractedTextType = Tuple[int, dict]
OCRInputType = Union[OCRImageType, ExtractedTextType]


def extract_features_from_block(block, page_height, page_width, all_font_sizes):
    spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
    if not spans:
        return None

    span = spans[0]
    text = span.get("text", "").strip()
    if not text:
        return None

    bbox = block.get("bbox", [0, 0, 0, 0])
    block_width = bbox[2] - bbox[0]
    block_height = bbox[3] - bbox[1]

    words = text.split()
    word_count = len(words)
    text_len = len(text)
    avg_word_len = text_len / word_count if word_count > 0 else 0
    font_size = span.get("size", 0.0)
    font_rank = sorted(list(set(all_font_sizes)), reverse=True).index(font_size) + 1 if font_size in all_font_sizes else 0

    return {
        "text": text,
        "text_len": text_len,
        "word_count": word_count,
        "line_count": len(block.get("lines", [])),
        "avg_word_len": avg_word_len,
        "font_size": font_size,
        "font_name": span.get("font", ""),
        "is_bold": int("bold" in span.get("font", "").lower()),
        "is_italic": int("italic" in span.get("font", "").lower()),
        "is_centered": int(abs((bbox[0] + bbox[2]) / 2 - page_width / 2) < page_width * 0.1),
        "is_uppercase": int(text.isupper()),
        "starts_with_number": int(text[:3].strip()[0].isdigit()) if text[:3].strip() else 0,
        "top_margin": bbox[1] / page_height,
        "bottom_margin": (page_height - bbox[3]) / page_height,
        "left_margin": bbox[0] / page_width,
        "right_margin": (page_width - bbox[2]) / page_width,
        "contains_colon": int(":" in text),
        "contains_bullet": int(text.strip().startswith(("-", "\u2022", "•"))),
        "special_char_count": sum(not c.isalnum() and not c.isspace() for c in text),
        "density_score": text_len / (block_width * block_height) if block_width * block_height > 0 else 0,
        "font_size_rank": font_rank,
        "text_length_bucket": "short" if text_len <= 20 else ("medium" if text_len <= 50 else "long"),
        "ends_with_colon": int(text.endswith(":")),
        "line_count_gt_3": int(len(block.get("lines", [])) > 3),
        "has_numbers_only": int(text.replace(".", "").isdigit()),
        "has_mixed_case": int(any(c.islower() for c in text) and any(c.isupper() for c in text)),
        "has_dash_or_underscore": int("-" in text or "_" in text),
        "is_all_punctuation": int(all(not c.isalnum() for c in text)),
        "word_density": word_count / text_len if text_len > 0 else 0,
    }


def _render_single_page(args: Tuple[str, int, int, int]) -> OCRInputType:
    pdf_path, page_num, total_pages, dpi_default = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)

    blocks = page.get_text("dict").get("blocks", [])
    font_sizes = [span.get("size", 0.0) for block in blocks for line in block.get("lines", []) for span in line.get("spans", [])]

    extracted = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        features = extract_features_from_block(block, page.rect.height, page.rect.width, font_sizes)
        if features:
            features["page_num"] = page_num
            features["is_first_page"] = int(page_num == 0)
            extracted.append(features)

    doc.close()

    if extracted:
        return (page_num, extracted)

    # Fallback to OCR
    dpi = dpi_default + 30 if page_num < total_pages * 0.1 else dpi_default - 30 if page_num > total_pages * 0.9 else dpi_default
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return (page_num, img)


def render_pages_to_images_parallel(pdf_path: str, dpi_default: int = 150):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    args_list = [(pdf_path, page_num, total_pages, dpi_default) for page_num in range(total_pages)]
    chunk_size = max(1, total_pages // 8)

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_render_single_page, args_list, chunksize=chunk_size))

    ocr_required_pages: List[OCRImageType] = []
    already_extracted_text: List[ExtractedTextType] = []

    for item in results:
        if isinstance(item[1], Image.Image):
            ocr_required_pages.append(item)
        else:
            already_extracted_text.append(item)

    return ocr_required_pages, already_extracted_text


def _ocr_single_image(args: OCRImageType) -> ExtractedTextType:
    page_num, image = args
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_image = Image.fromarray(thresh)

    text = pytesseract.image_to_string(preprocessed_image, config=TESSERACT_CONFIG)
    return (page_num, {"text": text.strip()})


def apply_ocr_to_images_parallel(image_data: List[OCRImageType]) -> List[ExtractedTextType]:
    if not image_data:
        return []
    total_pages = len(image_data)
    chunk_size = max(1, total_pages // 8)

    with ProcessPoolExecutor(max_workers=8) as executor:
        ocr_results = list(executor.map(_ocr_single_image, image_data, chunksize=chunk_size))

    return ocr_results


def main(pdf_path: str, output_json_path: str):
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    start = time.time()

    ocr_images, already_text = render_pages_to_images_parallel(pdf_path)
    ocr_results = apply_ocr_to_images_parallel(ocr_images)

    # Flatten the extracted structured text blocks
    flattened_blocks = [block for _, blocks in already_text for block in blocks]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(flattened_blocks, f, indent=2, ensure_ascii=False)

    # Optional: save OCR fallback too
    with open("sample_dataset/outputs/ocr_text_blocks.json", "w", encoding="utf-8") as f:
        json.dump({pg: block for pg, block in ocr_results}, f, indent=2, ensure_ascii=False)

    print("\n✅ Extraction completed.")
    print(f"🕒 Total time taken: {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    main("sample_dataset/pdfs/file02.pdf", "sample_dataset/outputs/structured_text_blocks.json")
