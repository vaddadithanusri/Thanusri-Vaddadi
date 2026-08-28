import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

TRAIN_PATH = Path("dataset/processed/train.csv")
TEST_PATH = Path("dataset/processed/test.csv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAIN FINAL GA-OPTIMIZED MODEL
# ============================================================

def train_ga_model():

    print("=" * 70)
    print("SuicideWatchAI - GA OPTIMIZED MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_df = pd.read_csv(TRAIN_PATH)

    print("Loading testing dataset...")

    test_df = pd.read_csv(TEST_PATH)

    print(f"\nTraining records: {len(train_df):,}")
    print(f"Testing records:  {len(test_df):,}")

    # --------------------------------------------------------
    # Extract text and labels
    # --------------------------------------------------------

    X_train_text = train_df["clean_text"].astype(str)
    y_train = train_df["label"].astype(int)

    X_test_text = test_df["clean_text"].astype(str)
    y_test = test_df["label"].astype(int)

    # --------------------------------------------------------
    # Load the SAME TF-IDF vectorizer used by baseline
    # --------------------------------------------------------

    print("\nLoading TF-IDF vectorizer...")

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    # --------------------------------------------------------
    # Transform text
    # --------------------------------------------------------

    print("\nTransforming training text...")

    X_train = vectorizer.transform(
        X_train_text
    )

    print("Transforming testing text...")

    X_test = vectorizer.transform(
        X_test_text
    )

    print(
        f"\nOriginal training matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Original testing matrix: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # Load GA-selected feature indices
    # --------------------------------------------------------

    selected_indices_path = (
        MODEL_DIR /
        "ga_selected_feature_indices.npy"
    )

    if not selected_indices_path.exists():

        print(
            "\nERROR: GA feature selection file "
            "was not found."
        )

        print(
            "Expected:"
        )

        print(selected_indices_path)

        return

    selected_indices = np.load(
        selected_indices_path
    )

    print(
        f"\nGA-selected features: "
        f"{len(selected_indices):,}"
    )

    # --------------------------------------------------------
    # Apply GA feature selection
    # --------------------------------------------------------

    print("\nApplying GA-selected features...")

    X_train_ga = X_train[
        :,
        selected_indices
    ]

    X_test_ga = X_test[
        :,
        selected_indices
    ]

    print(
        f"Optimized training matrix: "
        f"{X_train_ga.shape}"
    )

    print(
        f"Optimized testing matrix: "
        f"{X_test_ga.shape}"
    )

    # --------------------------------------------------------
    # Train optimized Linear SVM
    # --------------------------------------------------------

    print("\nTraining GA-optimized Linear SVM...")

    model = LinearSVC(
        C=1.0,
        random_state=42
    )

    model.fit(
        X_train_ga,
        y_train
    )

    print(
        "GA-optimized model training complete."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(
        X_test_ga
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GA OPTIMIZED MODEL RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Non-Suicide",
                "Suicide"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")

    print(cm)

    # --------------------------------------------------------
    # Feature reduction
    # --------------------------------------------------------

    original_features = X_train.shape[1]
    optimized_features = X_train_ga.shape[1]

    reduction_percentage = (
        (
            original_features -
            optimized_features
        )
        / original_features
    ) * 100

    print(
        f"\nOriginal features: "
        f"{original_features:,}"
    )

    print(
        f"Optimized features: "
        f"{optimized_features:,}"
    )

    print(
        f"Feature reduction: "
        f"{reduction_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Save optimized model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "ga_optimized_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nSaved GA-optimized model:"
    )

    print(model_path)

    # --------------------------------------------------------
    # Save selected features
    # --------------------------------------------------------

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    selected_feature_names = (
        feature_names[
            selected_indices
        ]
    )

    selected_features_path = (
        MODEL_DIR /
        "ga_selected_features.csv"
    )

    pd.DataFrame({
        "feature": selected_feature_names
    }).to_csv(
        selected_features_path,
        index=False
    )

    print(
        "\nSaved selected feature list:"
    )

    print(
        selected_features_path
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_path = (
        RESULT_DIR /
        "ga_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=[
            "Actual_Non_Suicide",
            "Actual_Suicide"
        ],
        columns=[
            "Predicted_Non_Suicide",
            "Predicted_Suicide"
        ]
    ).to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # Save GA model metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame({
        "Model": [
            "GA-Optimized TF-IDF + Linear SVM"
        ],
        "Features": [
            optimized_features
        ],
        "Accuracy": [
            accuracy
        ],
        "Precision": [
            precision
        ],
        "Recall": [
            recall
        ],
        "F1_Score": [
            f1
        ],
        "Feature_Reduction_Percent": [
            reduction_percentage
        ]
    })

    metrics_path = (
        RESULT_DIR /
        "ga_model_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        "\nSaved GA metrics:"
    )

    print(metrics_path)

    print(
        "\n" + "=" * 70
    )

    print(
        "GA MODEL TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_ga_model()