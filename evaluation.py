import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

METRICS_DIR = Path("results/metrics")
GRAPH_DIR = Path("results/graphs")

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL METRICS
# ============================================================

def load_metrics():

    print("=" * 70)
    print("SuicideWatchAI - MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading model metrics...")

    baseline_path = METRICS_DIR / "baseline_metrics.csv"
    ga_v1_path = METRICS_DIR / "ga_model_metrics.csv"
    ga_v2_path = METRICS_DIR / "ga_v2_model_metrics.csv"

    baseline = pd.read_csv(baseline_path)
    ga_v1 = pd.read_csv(ga_v1_path)
    ga_v2 = pd.read_csv(ga_v2_path)

    # --------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------

    baseline = baseline.rename(columns={
        "Model": "Model",
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1_Score": "F1-Score"
    })

    ga_v1 = ga_v1.rename(columns={
        "Model": "Model",
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1_Score": "F1-Score"
    })

    ga_v2 = ga_v2.rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score"
    })

    # --------------------------------------------------------
    # Select required columns
    # --------------------------------------------------------

    baseline_result = baseline[
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    ].copy()

    ga_v1_result = ga_v1[
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    ].copy()

    ga_v2_result = ga_v2[
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    ].copy()

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    comparison = pd.concat(
        [
            baseline_result,
            ga_v1_result,
            ga_v2_result
        ],
        ignore_index=True
    )

    return comparison


# ============================================================
# DISPLAY COMPARISON
# ============================================================

def display_comparison(comparison):

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print()

    display_df = comparison.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.4f}"
        )

    print(display_df.to_string(index=False))


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

def save_comparison(comparison):

    output_path = METRICS_DIR / "model_comparison.csv"

    comparison.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("Saved comparison table:")
    print(output_path)


# ============================================================
# PERFORMANCE GRAPH
# ============================================================

def create_performance_graph(comparison):

    models = comparison["Model"]

    x = range(len(models))
    width = 0.2

    plt.figure(figsize=(12, 7))

    plt.bar(
        [i - width * 1.5 for i in x],
        comparison["Accuracy"],
        width,
        label="Accuracy"
    )

    plt.bar(
        [i - width / 2 for i in x],
        comparison["Precision"],
        width,
        label="Precision"
    )

    plt.bar(
        [i + width / 2 for i in x],
        comparison["Recall"],
        width,
        label="Recall"
    )

    plt.bar(
        [i + width * 1.5 for i in x],
        comparison["F1-Score"],
        width,
        label="F1-Score"
    )

    plt.xticks(
        list(x),
        models,
        rotation=15,
        ha="right"
    )

    plt.ylabel("Score")
    plt.title("SuicideWatchAI - Model Performance Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output_path = GRAPH_DIR / "model_performance_comparison.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\nSaved performance graph:")
    print(output_path)


# ============================================================
# ACCURACY GRAPH
# ============================================================

def create_accuracy_graph(comparison):

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison["Model"],
        comparison["Accuracy"]
    )

    plt.ylabel("Accuracy")
    plt.title("SuicideWatchAI - Accuracy Comparison")
    plt.ylim(0, 1)

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.tight_layout()

    output_path = GRAPH_DIR / "accuracy_comparison.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\nSaved accuracy graph:")
    print(output_path)


# ============================================================
# F1 SCORE GRAPH
# ============================================================

def create_f1_graph(comparison):

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison["Model"],
        comparison["F1-Score"]
    )

    plt.ylabel("F1-Score")
    plt.title("SuicideWatchAI - F1-Score Comparison")
    plt.ylim(0, 1)

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.tight_layout()

    output_path = GRAPH_DIR / "f1_score_comparison.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\nSaved F1-score graph:")
    print(output_path)


# ============================================================
# FEATURE REDUCTION ANALYSIS
# ============================================================

def feature_reduction_analysis():

    print("\n" + "=" * 70)
    print("FEATURE REDUCTION ANALYSIS")
    print("=" * 70)

    original_features = 20000
    ga_v1_features = 500
    ga_v2_features = 561

    ga_v1_reduction = (
        (original_features - ga_v1_features)
        / original_features
    ) * 100

    ga_v2_reduction = (
        (original_features - ga_v2_features)
        / original_features
    ) * 100

    print(f"\nOriginal TF-IDF features : {original_features:,}")
    print(f"GA V1 selected features  : {ga_v1_features:,}")
    print(f"GA V2 selected features  : {ga_v2_features:,}")

    print(f"\nGA V1 feature reduction  : {ga_v1_reduction:.2f}%")
    print(f"GA V2 feature reduction  : {ga_v2_reduction:.2f}%")


# ============================================================
# MAIN
# ============================================================

def main():

    comparison = load_metrics()

    display_comparison(comparison)

    save_comparison(comparison)

    create_performance_graph(comparison)

    create_accuracy_graph(comparison)

    create_f1_graph(comparison)

    feature_reduction_analysis()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()