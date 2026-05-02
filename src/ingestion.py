import pandas as pd
import os

DATA_PATH = "data/raw"
OUTPUT_PATH = "data/processed"

def load_data():
    print("Loading datasets...")

    transaction = pd.read_csv(os.path.join(DATA_PATH, "train_transaction.csv"))
    identity = pd.read_csv(os.path.join(DATA_PATH, "train_identity.csv"))

    return transaction, identity


def merge_data(transaction, identity):
    print("Merging datasets...")

    df = transaction.merge(identity, on="TransactionID", how="left")

    print("Merged shape:", df.shape)

    return df


def save_data(df):
    print("Saving merged data...")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    file_path = os.path.join(OUTPUT_PATH, "merged_data.csv")
    df.to_csv(file_path, index=False)

    print(f"Saved to {file_path}")


if __name__ == "__main__":
    transaction, identity = load_data()
    df = merge_data(transaction, identity)
    save_data(df)