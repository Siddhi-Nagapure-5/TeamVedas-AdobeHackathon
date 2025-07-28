import os
import json
from pathlib import Path
from ocr import render_pages_to_images_parallel, apply_ocr_to_images_parallel
from logic.filter_blocks import filter_blocks
from logic.hybrid_classifier import classify_block
from logic.ml_model import load_model

# 📁 Directory paths
INPUT_DIR = Path("sample_dataset/pdfs")
OUTPUT_DIR = Path("sample_dataset/outputs")
OCR_JSON_PATH = OUTPUT_DIR / "ocr_text_blocks.json"
MODEL_PATH = "models/model1.pkl"

# ✅ Label filtering helper
ALLOWED_LABELS = {"Title", "H1", "H2", "H3", "H4"}

def flatten_blocks(structured_blocks):
    all_blocks = []
    for page_num, blocks in structured_blocks:
        for block in blocks:
            block["page_num"] = page_num
            all_blocks.append(block)
    return all_blocks

def classify_blocks(blocks):
    predictions = []
    for block in blocks:
        result = classify_block(block)
        if result is None:
            continue
        label = result["final_label"]
        if label not in ALLOWED_LABELS:
            continue
        predictions.append({
            "text": block.get("text", ""),
            "page_num": int(block.get("page_num", -1)),
            "selected_label": label,
            "source": result["source"],
            "heading_score": float(block.get("heading_score", 0.0)) if block.get("heading_score") else None,
            "candidates": result.get("candidates", [])
        })
    return sorted(predictions, key=lambda x: x["page_num"])

def process_pdf(pdf_path: Path):
    print(f"\n📄 Processing: {pdf_path.name}")
    base_name = pdf_path.stem
    structured_output_path = OUTPUT_DIR / f"{base_name}_output.json"

    # Step 1: Extract structured blocks from PDF
    ocr_pages, structured_blocks = render_pages_to_images_parallel(str(pdf_path))
    ocr_results = apply_ocr_to_images_parallel(ocr_pages)

    # Step 2: Flatten + filter
    all_blocks = flatten_blocks(structured_blocks)
    print(f"🔍 Extracted {len(all_blocks)} text blocks.")

    filtered_blocks = filter_blocks(all_blocks)
    print(f"✅ {len(filtered_blocks)} blocks passed filtering.")

    # Step 3: Classify
    predictions = classify_blocks(filtered_blocks)

    # Step 3.5: Convert predictions to structured outline format
    document_title_parts = [
        block["text"].strip() for block in predictions
        if block["selected_label"] == "Title" and block["page_num"] <= 1
    ]
    document_title = " ".join(document_title_parts).strip()

    outline = []
    for block in predictions:
        label = block["selected_label"]
        if label in {"H1", "H2", "H3", "H4"}:
            outline.append({
                "level": label,
                "text": block["text"].strip(),
                "page": block["page_num"] + 1
            })

    structured_output = {
        "title": document_title,
        "outline": outline
    }

    # Step 4: Save output
    os.makedirs(structured_output_path.parent, exist_ok=True)
    with open(structured_output_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2, ensure_ascii=False)
    print(f"📂 Saved predictions to: {structured_output_path.name}")

    return base_name, {pg: val for pg, val in ocr_results}

def process_all_pdfs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_model(MODEL_PATH)
    print("📊 Model loaded.\n")

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in:", INPUT_DIR.resolve())
        return

    print(f"🚀 Found {len(pdf_files)} PDF(s) to process.")
    ocr_fallback_all = {}

    for pdf_path in pdf_files:
        name, fallback = process_pdf(pdf_path)
        if fallback:
            ocr_fallback_all[name] = fallback

    # Optional: Save fallback OCR blocks
    with open(OCR_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ocr_fallback_all, f, indent=2, ensure_ascii=False)
    print(f"📑 OCR fallback saved to: {OCR_JSON_PATH.name}")

if __name__ == "__main__":
    print("🔥 Starting Challenge_1a Heading Classification Pipeline")
    process_all_pdfs()
