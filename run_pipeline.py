import os
import json
from logic.filter_blocks import filter_blocks
from logic.ml_model import load_model
from logic.hybrid_classifier import classify_block

INPUT_JSON_PATH = "sample_dataset/outputs/structured_text_blocks.json"
OUTPUT_JSON_PATH = "sample_dataset/outputs/file02_output.json"
MODEL_PATH = "models/model1.pkl"

# ✅ Custom label mapping (used only for ML labels)
custom_label_map = {
    0: "Title",
    1: "H1",
    2: "H3",
    3: "H2",
    4: "H4",
    5: "H5"  # will be skipped
}

def flatten_blocks(json_data) -> list:
    all_blocks = []
    if isinstance(json_data, dict):
        for page_num, blocks in json_data.items():
            for block in blocks:
                block["page_num"] = int(page_num)
                all_blocks.append(block)
    elif isinstance(json_data, list):
        for block in json_data:
            if "page_num" not in block:
                block["page_num"] = -1
            all_blocks.append(block)
    else:
        raise ValueError("Invalid structured_text_blocks format.")
    return all_blocks

def run_pipeline():
    print("🚀 Running heading classification pipeline...\n")

    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_JSON_PATH}")
    
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        structured_data = json.load(f)
    
    all_blocks = flatten_blocks(structured_data)
    print(f"📦 Loaded {len(all_blocks)} blocks from input.")

    filtered_blocks = filter_blocks(all_blocks)
    print(f"✅ {len(filtered_blocks)} blocks passed advanced filtering.\n")

    load_model(MODEL_PATH)
    print("📊 ML model loaded.\n")

    predictions = []
    for block in filtered_blocks:
        result = classify_block(block)

        if result is None:
            continue  # Skip blocks filtered as H5

        raw_label = result["final_label"]

        # Map ML numeric labels to string using custom mapping
        if result["source"] == "ml":
            try:
                label_num = int(raw_label)
                mapped_label = custom_label_map.get(label_num, str(label_num))
            except:
                mapped_label = str(raw_label)
        else:
            # Manual rule-based label
            mapped_label = str(raw_label)

        # ❌ Skip if mapped label is H5 or unknown
        if mapped_label == "H5" or mapped_label not in ["Title", "H1", "H2", "H3", "H4"]:
            continue

        predictions.append({
            "text": str(block.get("text", "")),
            "page_num": int(block.get("page_num", -1)),
            "selected_label": mapped_label,
            "source": str(result["source"]),
            "heading_score": float(block.get("heading_score", 0.0)) if block.get("heading_score") is not None else None,
            "candidates": [str(c) for c in result["candidates"]]
        })

    # ✅ Sort results by page number
    predictions.sort(key=lambda x: x["page_num"])

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ Classification completed. Output saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    run_pipeline()
