from fastapi import FastAPI
from pydantic import BaseModel
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import random

app = FastAPI()

# -----------------------------
# Metrics
# -----------------------------
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
REQUEST_LATENCY = Histogram("api_latency_seconds", "API Latency")
FRAUD_PREDICTIONS = Counter("fraud_predictions_total", "Fraud Predictions")
NON_FRAUD_PREDICTIONS = Counter("non_fraud_predictions_total", "Non-Fraud Predictions")

# -----------------------------
# Input schema
# -----------------------------
class InputData(BaseModel):
    feature1: float
    feature2: float

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InputData):
    start_time = time.time()
    REQUEST_COUNT.inc()

    # 🔥 Replace with real model later
    prediction = random.choice([0, 1])

    if prediction == 1:
        FRAUD_PREDICTIONS.inc()
    else:
        NON_FRAUD_PREDICTIONS.inc()

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"prediction": int(prediction)}

# -----------------------------
# Metrics endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")