import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from mlflow_config import setup_mlflow
import os
import sys

DATA_PATH = "data/processed/final_data.csv"

if not os.path.exists(DATA_PATH):
    print("Dataset not found. Skipping training...")
    sys.exit(0)

# -----------------------------
# Setup MLflow
# -----------------------------
setup_mlflow()

# -----------------------------
# Load Data
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("data/processed/final_data.csv")

y = df["isFraud"]
X = df.drop("isFraud", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Class Weight
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -----------------------------
# STEP 1 — Train base model
# -----------------------------
print("Training base XGBoost...")

base_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

base_model.fit(X_train, y_train)

# -----------------------------
# STEP 2 — Feature Importance
# -----------------------------
importances = base_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Select top features
TOP_K = 50
top_features = importance_df.head(TOP_K)["feature"].tolist()

print(f"Selected top {TOP_K} features")

# Reduce dataset
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]

# -----------------------------
# STEP 3 — Retrain on selected features
# -----------------------------
with mlflow.start_run(run_name="XGBoost_Hybrid_FeatureSelection"):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    print("Training hybrid model...")
    model.fit(X_train_sel, y_train)

    # Predictions
    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nHybrid Model Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    # MLflow logging
    mlflow.log_param("model", "XGBoost_Hybrid")
    mlflow.log_param("top_features", TOP_K)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)

    mlflow.xgboost.log_model(model, "model")

print("Hybrid model training complete!")