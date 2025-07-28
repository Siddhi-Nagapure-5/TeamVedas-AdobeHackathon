import joblib
import os
import numpy as np
from typing import Dict, List

MODEL = None
COLUMNS = []

# ✅ Label mapping (must be consistent with run_pipeline.py)
LABEL_MAP = {
    0: "Title",
    1: "H1",
    2: "H2",
    3: "H3",
    4: "H4",
    5: "H5",  # We'll exclude this from output
}

ALLOWED_LABELS = {"Title", "H1", "H2", "H3", "H4"}  # ✅ Final output should only include these


def load_model(model_path: str):
    global MODEL, COLUMNS
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"📦 Loading model from: {model_path}")
    MODEL = joblib.load(model_path)
    if hasattr(MODEL, "feature_names_in_"):
        COLUMNS = MODEL.feature_names_in_.tolist()
    else:
        raise Exception("Model missing feature names info!")


def predict_label_with_confidence(block: Dict) -> List[Dict]:
    global MODEL, COLUMNS
    if MODEL is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    input_vector = np.array([[block.get(col, 0) for col in COLUMNS]])
    probs = MODEL.predict_proba(input_vector)[0]
    labels = MODEL.classes_

    results = []
    for label, conf in zip(labels, probs):
        label_str = LABEL_MAP.get(label, str(label))  # Convert label index to readable name
        if label_str in ALLOWED_LABELS:
            results.append({
                "label": label_str,
                "confidence": round(float(conf), 3)
            })

    sorted_results = sorted(results, key=lambda x: -x["confidence"])
    return sorted_results
