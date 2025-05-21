import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load datasets
fake_df = pd.read_csv("fake.csv")
real_df = pd.read_csv("Real.csv")

# Assign labels
fake_df["label"] = 0  # Fake news
real_df["label"] = 1  # Real news

# Merge datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Select necessary columns
df = df[["text", "label"]].dropna()

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Convert to Hugging Face dataset format
train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_data = Dataset.from_dict({"text": val_texts, "label": val_labels})
