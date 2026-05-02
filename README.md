# Fraud Detection MLOps Pipeline

## Overview

This project implements a full MLOps pipeline for fraud detection using the IEEE-CIS dataset. The system is designed to maintain high recall, handle class imbalance, and monitor performance in real time.

---

## Pipeline Components

### 1. Data Pipeline

* Data Ingestion (merge transaction + identity)
* Data Validation (missing handling, cleaning)
* Feature Engineering & Encoding

### 2. Models

* XGBoost (primary model with class weights)
* LightGBM (comparison model)
* Hybrid Model (feature selection + XGBoost)

### 3. Imbalance Handling

* Cost-sensitive learning using `scale_pos_weight`
* Justification: avoids synthetic noise from SMOTE

---

## CI/CD (GitHub Actions)

* Trigger on push / PR
* Runs validation and training pipeline
* Ensures pipeline integrity

---

## Monitoring System

### Prometheus

Tracks:

* API request count
* Latency
* Fraud predictions

### Grafana Dashboards

* System Health Dashboard
* Model Performance Dashboard

---

## Alerts

Configured alerts for:

* High latency
* Low fraud detection rate
* High traffic

---

## Intelligent Retraining

* Alerts simulate triggering retraining
* `retrain.py` executes model training

---

## Tech Stack

* Python, FastAPI
* XGBoost, LightGBM
* MLflow
* Prometheus, Grafana
* GitHub Actions

---

## Conclusion

The system demonstrates a production-style MLOps pipeline with monitoring, alerting, and automated retraining strategies for fraud detection.
