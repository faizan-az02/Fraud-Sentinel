import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
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

print("Train shape:", X_train.shape)

# -----------------------------
# Class Weight
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -----------------------------
# MLflow Run
# -----------------------------
with mlflow.start_run(run_name="LightGBM_ClassWeight"):

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    print("Training LightGBM...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nLightGBM Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    # MLflow logging
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)

    mlflow.lightgbm.log_model(model, "model")

print("LightGBM training complete!")