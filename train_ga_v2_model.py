import numpy as np
import pandas as pd
import joblib

from pathlib import Path

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
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# MAIN
# ============================================================

def train_ga_v2_model():

    print("=" * 70)
    print("SuicideWatchAI - GA V2 OPTIMIZED MODEL")
    print("=" * 70)


    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_df = pd.read_csv(TRAIN_PATH)

    print("Loading testing dataset...")

    test_df = pd.read_csv(TEST_PATH)


    print(
        f"\nTraining records: "
        f"{len(train_df):,}"
    )

    print(
        f"Testing records:  "
        f"{len(test_df):,}"
    )


    # --------------------------------------------------------
    # Load TF-IDF vectorizer
    # --------------------------------------------------------

    print("\nLoading TF-IDF vectorizer...")

    vectorizer = joblib.load(
        MODEL_DIR /
        "tfidf_vectorizer.joblib"
    )


    # --------------------------------------------------------
    # Load GA V2 selected features
    # --------------------------------------------------------

    print(
        "\nLoading GA V2 selected features..."
    )

    selected_features = np.load(
        MODEL_DIR /
        "ga_v2_selected_feature_indices.npy"
    )


    print(
        f"GA V2-selected features: "
        f"{len(selected_features):,}"
    )


    # --------------------------------------------------------
    # Transform training text
    # --------------------------------------------------------

    print(
        "\nTransforming training text..."
    )

    X_train_full = vectorizer.transform(
        train_df["clean_text"].astype(str)
    )


    # --------------------------------------------------------
    # Transform testing text
    # --------------------------------------------------------

    print(
        "Transforming testing text..."
    )

    X_test_full = vectorizer.transform(
        test_df["clean_text"].astype(str)
    )


    print(
        f"\nOriginal training matrix: "
        f"{X_train_full.shape}"
    )

    print(
        f"Original testing matrix: "
        f"{X_test_full.shape}"
    )


    # --------------------------------------------------------
    # Apply GA V2 features
    # --------------------------------------------------------

    print(
        "\nApplying GA V2-selected features..."
    )


    X_train = X_train_full[
        :,
        selected_features
    ]


    X_test = X_test_full[
        :,
        selected_features
    ]


    print(
        f"GA V2 training matrix: "
        f"{X_train.shape}"
    )

    print(
        f"GA V2 testing matrix: "
        f"{X_test.shape}"
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_train = train_df[
        "label"
    ].astype(int).values


    y_test = test_df[
        "label"
    ].astype(int).values


    # --------------------------------------------------------
    # Train final SVM
    # --------------------------------------------------------

    print(
        "\nTraining GA V2-optimized Linear SVM..."
    )


    model = LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE
    )


    model.fit(
        X_train,
        y_train
    )


    print(
        "GA V2-optimized model training complete."
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )


    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GA V2 MODEL RESULTS")
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


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )


    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Non-Suicide",
            "Suicide"
        ],
        zero_division=0
    )


    print(report)


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print(
        "Confusion Matrix:"
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    print(cm)


    # --------------------------------------------------------
    # Feature reduction
    # --------------------------------------------------------

    original_feature_count = (
        X_train_full.shape[1]
    )


    optimized_feature_count = (
        X_train.shape[1]
    )


    reduction_percentage = (
        1 -
        (
            optimized_feature_count /
            original_feature_count
        )
    ) * 100


    print(
        f"\nOriginal features: "
        f"{original_feature_count:,}"
    )


    print(
        f"Optimized features: "
        f"{optimized_feature_count:,}"
    )


    print(
        f"Feature reduction: "
        f"{reduction_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "ga_v2_optimized_svm.joblib"
    )


    joblib.dump(
        model,
        model_path
    )


    print(
        "\nSaved GA V2 model:"
    )

    print(
        model_path
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "model": "GA V2 Optimized Linear SVM",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "original_features": original_feature_count,
            "optimized_features": optimized_feature_count,
            "feature_reduction_percent":
                reduction_percentage
        }
    ])


    metrics_path = (
        RESULT_DIR /
        "ga_v2_model_metrics.csv"
    )


    metrics_df.to_csv(
        metrics_path,
        index=False
    )


    print(
        "\nSaved GA V2 metrics:"
    )

    print(
        metrics_path
    )


    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        cm,
        index=[
            "Actual Non-Suicide",
            "Actual Suicide"
        ],
        columns=[
            "Predicted Non-Suicide",
            "Predicted Suicide"
        ]
    )


    confusion_path = (
        RESULT_DIR /
        "ga_v2_confusion_matrix.csv"
    )


    confusion_df.to_csv(
        confusion_path
    )


    print(
        "Saved GA V2 confusion matrix:"
    )

    print(
        confusion_path
    )


    print("\n" + "=" * 70)
    print("GA V2 MODEL TRAINING COMPLETE")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_ga_v2_model()