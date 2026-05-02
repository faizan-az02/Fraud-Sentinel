import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import sys

INPUT_PATH = "data/processed/merged_data.csv"
OUTPUT_PATH = "data/processed/final_data.csv"


def load_data():
    print("Loading merged data...")
    if not os.path.exists(INPUT_PATH):
        print("Dataset not found. Skipping training...")
        sys.exit(0)
    df = pd.read_csv(INPUT_PATH)

    print("Shape:", df.shape)
    return df


# -----------------------------
# Step 1 — Drop high missing columns
# -----------------------------
def drop_high_missing(df, threshold=80):
    print(f"Dropping columns with >{threshold}% missing values...")

    missing = df.isnull().mean() * 100
    cols_to_drop = missing[missing > threshold].index

    df = df.drop(columns=cols_to_drop)

    print(f"Dropped {len(cols_to_drop)} columns")
    print("New shape:", df.shape)

    return df


# -----------------------------
# Step 2 — Fill missing values
# -----------------------------
def fill_missing(df):
    print("Filling missing values...")

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("missing")

    print("Missing values after fill:", df.isnull().sum().sum())

    return df


# -----------------------------
# Step 3 — Drop useless columns
# -----------------------------
def drop_useless(df):
    print("Dropping ID/time columns...")

    cols_to_drop = ["TransactionID", "TransactionDT"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# -----------------------------
# Step 4 — Encode categorical features
# -----------------------------
def encode_features(df):
    print("Encoding categorical features...")

    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    print(f"Encoded {len(cat_cols)} columns")

    return df


# -----------------------------
# Step 5 — Final checks
# -----------------------------
def validate(df):
    print("Running final validation checks...")

    # No missing values
    assert df.isnull().sum().sum() == 0, "❌ Missing values still exist"

    # No object columns
    assert len(df.select_dtypes(include="object").columns) == 0, "❌ Object columns still exist"

    print("✅ Validation passed")


# -----------------------------
# Step 6 — Save
# -----------------------------
def save_data(df):
    print("Saving final dataset...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to {OUTPUT_PATH}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    df = load_data()

    df = drop_high_missing(df)
    df = fill_missing(df)
    df = drop_useless(df)
    df = encode_features(df)

    validate(df)
    save_data(df)