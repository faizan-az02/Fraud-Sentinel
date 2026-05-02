import pandas as pd
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

# Split features & target
y = df["isFraud"]
X = df.drop("isFraud", axis=1)

# Train-test split (IMPORTANT: stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# Handle Imbalance (Class Weight)
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# -----------------------------
# MLflow Run
# -----------------------------
with mlflow.start_run(run_name="XGBoost_ClassWeight"):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nModel Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    # -----------------------------
    # MLflow Logging
    # -----------------------------
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)

    # Log model
    mlflow.xgboost.log_model(model, "model")

print("Training complete!")