import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# PATH CONFIGURATION
# ============================================================

DATASET_PATH = Path("dataset/raw/Suicide_Detection.csv")

PROCESSED_DIR = Path("dataset/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    """
    Clean a social-media text while preserving
    words that may contain meaningful signals.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove usernames such as @user
    text = re.sub(r"@\w+", " ", text)

    # Keep words and basic apostrophes
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_dataset():

    print("=" * 70)
    print("SuicideWatchAI - Dataset Preprocessing")
    print("=" * 70)

    if not DATASET_PATH.exists():
        print("\nERROR: Dataset not found.")
        print(DATASET_PATH)
        return

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(f"Original records: {len(df):,}")

    # --------------------------------------------------------
    # Remove unnecessary index column
    # --------------------------------------------------------

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna(subset=["text", "class"])

    # --------------------------------------------------------
    # Remove duplicate records
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(subset=["text"])

    duplicates_removed = before_duplicates - len(df)

    print(f"Duplicates removed: {duplicates_removed:,}")

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    print("\nCleaning text...")

    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty texts after cleaning
    df = df[df["clean_text"].str.strip() != ""]

    # --------------------------------------------------------
    # Encode target labels
    # --------------------------------------------------------

    df["label"] = df["class"].map({
        "non-suicide": 0,
        "suicide": 1
    })

    # Remove unexpected labels if any
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # --------------------------------------------------------
    # Display final dataset information
    # --------------------------------------------------------

    print("\nFinal dataset:")
    print(f"Records: {len(df):,}")

    print("\nClass distribution:")

    print(df["class"].value_counts())

    print("\nEncoded class distribution:")

    print(df["label"].value_counts())

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    print("\nTrain/Test Split:")
    print(f"Training records: {len(train_df):,}")
    print(f"Testing records: {len(test_df):,}")

    # --------------------------------------------------------
    # Save processed datasets
    # --------------------------------------------------------

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved files:")

    print(f"Training dataset: {train_path}")
    print(f"Testing dataset:  {test_path}")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preprocess_dataset()