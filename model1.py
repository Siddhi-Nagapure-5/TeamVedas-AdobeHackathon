import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# Load and preprocess dataset
# ------------------------------
df = pd.read_csv("model1.csv")
df = df.dropna()

# Drop raw text column if present
if "text" in df.columns:
    df = df.drop(columns=["text"])

# Encode target
target_encoder = LabelEncoder()
df["label"] = target_encoder.fit_transform(df["label"])

# One-hot encode categorical features
categorical_cols = ["font_name", "text_length_bucket"]
df = pd.get_dummies(df, columns=categorical_cols)

# Split features and labels
X = df.drop(columns=["label"])
y = df["label"]

# ✅ Ensure all data is float type (SHAP fix)
X = X.astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# Train XGBoost model
# ------------------------------
model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(target_encoder.classes_),
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    eval_metric="mlogloss",
    use_label_encoder=False,
    verbosity=0
)
model.fit(X_train, y_train)

# ------------------------------
# Save model and encoder
# ------------------------------
os.makedirs("models", exist_ok=True)
with open("models/model1.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

print("✅ Model and label encoder saved.")

# ------------------------------
# Predict for sample block
# ------------------------------
new_block_data = {
    "bold": 0,
    "uppercase": 0,
    "is_numbered_list": 0,
    "width": 0.7532907373764935 - 0.11592156902637357,
    "height": 0.6729846530490451 - 0.3072494352706755,
    "x": 0.11592156902637357,
    "y": 0.3072494352706755,
    "font_size": 14.039999961853027,
    "font_size_rank": 2,
    "text_len": 11,
    "word_count": 2,
    "avg_word_len": 5.5,
    "line_count": 1,
    "is_italic": 0,
    "is_centered": 0,
    "starts_with_number": 1,
    "contains_colon": 0,
    "contains_bullet": 0,
    "special_char_count": 1,
    "density_score": 0.008778743295668745,
    "ends_with_colon": 0,
    "line_count_gt_3": 0,
    "has_numbers_only": 0,
    "has_mixed_case": 1,
    "has_dash_or_underscore": 0,
    "is_all_punctuation": 0,
    "word_density": 0.18181818181818182,
    "text_length_bucket": "short",
    "page_num": 9,
    "is_first_page": 0,
    "font_name": "Arial"
}

# Prepare sample input
sample_df = pd.DataFrame([new_block_data])
sample_df = pd.get_dummies(sample_df)

# Align with training columns
for col in X.columns:
    if col not in sample_df.columns:
        sample_df[col] = 0
sample_df = sample_df[X.columns]

# Predict
pred = model.predict(sample_df)
predicted_label = target_encoder.inverse_transform(pred)[0]
print("🔎 Predicted Heading Level (sample):", predicted_label)

# ------------------------------
# Evaluation
# ------------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(
    target_encoder.inverse_transform(y_test),
    target_encoder.inverse_transform(y_pred)
))

# Confusion Matrix
cm = confusion_matrix(
    target_encoder.inverse_transform(y_test),
    target_encoder.inverse_transform(y_pred),
    labels=target_encoder.classes_
)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_fixed.png")
plt.show()

# ------------------------------
# SHAP Plot
# ------------------------------
print("\n📌 Generating SHAP summary plot (first 100 samples)...")
explainer = shap.Explainer(model, X_train[:100])
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], plot_type="bar")
