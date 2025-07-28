from typing import Dict, Optional
from logic.manual_rules import apply_manual_rules
from logic.ml_model import predict_label_with_confidence

# ✅ Mapping from numeric class to heading labels
label_map = {
    0: "Title",
    1: "H1",
    2: "H2",
    3: "H3",
    4: "H4",
    5: "H5"
}

# ✅ Ignore these labels completely
SKIP_LABELS = {"H5"}


def classify_block(block: Dict) -> Optional[Dict]:
    """
    First apply manual rules. If that fails, fallback to ML model.
    Skips blocks classified as H5.
    """
    manual_result = apply_manual_rules(block)

    if manual_result:
        if manual_result in SKIP_LABELS:
            return None
        return {
            "final_label": manual_result,
            "source": "manual",
            "candidates": []
        }

    # ML fallback
    ml_results = predict_label_with_confidence(block)

    if not ml_results:
        return None

    # Extract top prediction
    top_label = ml_results[0]["label"]
    mapped_label = label_map.get(int(top_label), str(top_label)) if isinstance(top_label, (int, float)) else str(top_label)

    if mapped_label in SKIP_LABELS:
        return None

    # Map candidates
    mapped_candidates = []
    for candidate in ml_results:
        label = candidate.get("label")
        if isinstance(label, (int, float)):
            candidate["label"] = label_map.get(int(label), str(label))
        else:
            candidate["label"] = str(label)
        mapped_candidates.append(str(candidate))  # 👈 serialize to string

    return {
        "final_label": mapped_label,
        "source": "ml",
        "candidates": mapped_candidates
    }
