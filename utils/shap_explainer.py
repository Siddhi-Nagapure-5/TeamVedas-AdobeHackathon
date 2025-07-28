# utils/shap_explainer.py

import shap
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from typing import Dict, List


def load_model_and_features(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model does not contain feature names!")

    return model, list(model.feature_names_in_)


def explain_single_prediction(model, feature_names: List[str], block: Dict):
    """
    Explains the model's prediction for a single block using SHAP waterfall plot.
    Supports multi-class XGBoost.
    """
    # Convert input block to instance array
    instance = np.array([[float(block.get(f, 0)) for f in feature_names]])

    # Predict the class
    predicted_class = model.predict(instance)[0]

    # SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(instance)

    # Extract values for predicted class
    class_shap_values = shap_values.values[0][predicted_class]  # shape: (n_features,)
    class_base_value = shap_values.base_values[0][predicted_class]  # scalar
    class_data = shap_values.data[0]  # input values (same for all classes)

    # ✅ Build new Explanation object manually
    explanation = shap.Explanation(
        values=class_shap_values,
        base_values=class_base_value,
        data=class_data,
        feature_names=feature_names
    )

    print(f"\n🧠 Explaining prediction for: \"{block.get('text', '')[:50]}...\"")
    print(f"🔎 Predicted class index: {predicted_class}")

    # Show waterfall
    shap.plots.waterfall(explanation, show=True)


def explain_global_feature_importance(model, X_sample: np.ndarray, feature_names: List[str]):
    """
    Shows SHAP global feature importance using bar/summary plot.
    """
    print("\n📊 Generating SHAP summary plot (global feature importance)...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names)


if __name__ == "__main__":
    MODEL_PATH = "models/model1.pkl"
    STRUCTURED_PATH = "data/output/structured_text_blocks.json"

    model, features = load_model_and_features(MODEL_PATH)

    with open(STRUCTURED_PATH, "r", encoding="utf-8") as f:
        structured = json.load(f)

    # 📦 Handle both dict and list structure
    if isinstance(structured, dict):
        blocks = [b for page in structured.values() for b in page][:10]
    elif isinstance(structured, list):
        blocks = structured[:10]
    else:
        raise ValueError("Invalid format for structured_text_blocks.json")

    # Convert each block to numeric values (safe for SHAP)
    X_sample = np.array([
        [float(b.get(f, 0)) for f in features]
        for b in blocks
    ])

    # SHAP summary plot (global)
    explain_global_feature_importance(model, X_sample, features)

    # Explain one prediction (with correct class extraction)
    explain_single_prediction(model, features, blocks[0])
