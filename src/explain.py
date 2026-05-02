import pandas as pd
import shap
import xgboost as xgb
from mlflow_config import setup_mlflow
import mlflow

# Setup MLflow
setup_mlflow()

# ---------
# Load data
# ---------
df = pd.read_csv("data/processed/final_data.csv")

y = df["isFraud"]
X = df.drop("isFraud", axis=1)

# -----------
# Train model
# -----------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X, y)

# --------------
# SHAP Explainer
# --------------
print("Generating SHAP values...")

explainer = shap.Explainer(model)
shap_values = explainer(X)

# ------------
# Summary plot
# ------------
shap.summary_plot(shap_values, X)

# -------------
# Log to MLflow
# -------------
with mlflow.start_run(run_name="SHAP_Explainability"):
    mlflow.log_param("explainability", "SHAP")