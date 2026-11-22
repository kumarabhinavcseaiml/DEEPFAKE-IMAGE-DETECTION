"""Train three classical ML models for real-vs-fake face image classification.

Expected folders (images may also be nested below each class folder):
    data/raw/real/
    data/raw/fake/

Run from this project's root folder:
    python train.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.feature import hog

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

IMAGE_SIZE = (64, 64)
HOG_SETTINGS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
def list_image_files(folder: Path) -> list[Path]:
    """Return supported image files recursively from one class folder."""
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def load_image_and_extract_hog(image_path: Path) -> np.ndarray | None:
    """Read an image, resize it, convert to grayscale and compute HOG features."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    vector = hog(gray, **HOG_SETTINGS)
    return np.asarray(vector, dtype=np.float32)


def build_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Build feature matrix and labels from data/raw/real and data/raw/fake."""
    if not REAL_DIR.exists() or not FAKE_DIR.exists():
        raise FileNotFoundError(
            "Dataset folders were not found. Required folders are:\n"
            f"  {REAL_DIR}\n  {FAKE_DIR}\n"
            "Place real face images in data/raw/real and fake face images in data/raw/fake."
        )

    class_files = {
        "real": list_image_files(REAL_DIR),
        "fake": list_image_files(FAKE_DIR),
    }
    print("\nDataset files found:")
    print(f"  REAL: {len(class_files['real']):,}")
    print(f"  FAKE: {len(class_files['fake']):,}")

    if not class_files["real"] or not class_files["fake"]:
        raise ValueError(
            "Both data/raw/real and data/raw/fake must contain image files."
        )

    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    manifest_rows: list[dict] = []
    total_files = sum(len(paths) for paths in class_files.values())
    processed = 0
    skipped = 0

    print("\nSTEP 1/6 — Preprocessing images and extracting HOG features")
    for label, paths in class_files.items():
        for image_path in paths:
            processed += 1
            features = load_image_and_extract_hog(image_path)
            if features is None:
                skipped += 1
                continue

            feature_rows.append(features)
            labels.append(label)
            manifest_rows.append({"path": str(image_path.relative_to(ROOT)), "label": label})

            if processed % 250 == 0 or processed == total_files:
                print(f"  Processed {processed:,}/{total_files:,} files")

    if len(set(labels)) != 2:
        raise ValueError(
            "Could not read images from both classes. Check that the folders contain valid JPG/PNG images."
        )

    counts = pd.Series(labels).value_counts().to_dict()
    if min(counts.get("real", 0), counts.get("fake", 0)) < 5:
        raise ValueError("At least five readable images per class are required for a stratified split.")

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.asarray(labels, dtype=str)
    manifest = pd.DataFrame(manifest_rows)
    summary = {
        "total_images": int(len(y)),
        "real_images": int(counts.get("real", 0)),
        "fake_images": int(counts.get("fake", 0)),
        "skipped_unreadable": int(skipped),
        "image_size": list(IMAGE_SIZE),
        "feature_method": "HOG",
    }
    return X, y, manifest, summary
def save_eda(summary: dict, manifest: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(REPORTS_DIR / "dataset_manifest.csv", index=False)

    with (REPORTS_DIR / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    sns.set_theme(style="whitegrid")
    counts = pd.Series({"REAL": summary["real_images"], "FAKE": summary["fake_images"]})
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(counts.index, counts.values, color=["#238B74", "#E07A5F"], width=0.55)
    ax.set_title("Dataset Class Distribution", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("Image category")
    ax.set_ylabel("Number of readable images")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, padding=4, fmt="%d", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for row, label in enumerate(("real", "fake")):
        sample_paths = list_image_files(REAL_DIR if label == "real" else FAKE_DIR)[:4]
        for col in range(4):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(sample_paths):
                image = cv2.imread(str(sample_paths[col]))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image)
            if col == 0:
                ax.set_ylabel(label.upper(), fontsize=11, weight="bold")
    fig.suptitle("Sample images from the dataset", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "sample_images.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
def create_models() -> dict[str, Pipeline]:
    """Create three comparable classical ML pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=220,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def positive_class_probability(model: Pipeline, X: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    if "fake" not in classes:
        return None
    probabilities = model.predict_proba(X)
    return probabilities[:, classes.index("fake")]


def evaluate_one_model(
    name: str,
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    predictions = model.predict(X_test)
    fake_probabilities = positive_class_probability(model, X_test)

    metrics = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, pos_label="fake", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, pos_label="fake", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, pos_label="fake", zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test == "fake", fake_probabilities)) if fake_probabilities is not None else float("nan"),
    }

    safe_name = name.lower().replace(" ", "_")
    report = classification_report(
        y_test,
        predictions,
        labels=["real", "fake"],
        target_names=["REAL", "FAKE"],
        zero_division=0,
    )
    (REPORTS_DIR / f"{safe_name}_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, predictions, labels=["real", "fake"])
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(f"{name} — Confusion Matrix", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{safe_name}_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return metrics


def save_model_comparison(results: pd.DataFrame, roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> None:
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    plot_data = results.set_index("Model")[metric_columns] * 100
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_data.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Performance comparison of ML algorithms", fontsize=15, weight="bold", pad=14)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric", frameon=False, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if roc_data:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for name, (fpr, tpr, auc_value) in roc_data.items():
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_value:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
        ax.set_title("ROC curves on the test set", fontsize=15, weight="bold", pad=14)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "roc_curves.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 58)
    print("       DEEPFAKE IMAGE DETECTION — MODEL TRAINING")
    print("=" * 58)

    X, y, manifest, summary = build_dataset()
    save_eda(summary, manifest)
    print(f"\nReadable images: {summary['total_images']:,} | Skipped unreadable: {summary['skipped_unreadable']:,}")
    print(f"HOG feature matrix: {X.shape}")

    print("\nSTEP 2/6 — Creating stratified train / validation / test splits")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_train_val
    )
    print(f"  Training:   {len(y_train):,} images (60%)")
    print(f"  Validation: {len(y_val):,} images (20%)")
    print(f"  Testing:    {len(y_test):,} images (20%)")

    models = create_models()
    validation_scores: dict[str, float] = {}
    fitted_models: dict[str, Pipeline] = {}
    test_results: list[dict] = []
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    print("\nSTEP 3/6 — Training and validating three ML algorithms")
    for name, model in models.items():
        print(f"\n  Training {name}...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*deprecated.*")
            model.fit(X_train, y_train)

        fitted_models[name] = model
        val_predictions = model.predict(X_val)
        validation_f1 = f1_score(y_val, val_predictions, pos_label="fake", zero_division=0)
        validation_scores[name] = float(validation_f1)
        print(f"  Validation F1 (FAKE class): {validation_f1:.4f}")

    print("\nSTEP 4/6 — Evaluating models on the held-out test set")
    for name, model in fitted_models.items():
        metrics = evaluate_one_model(name, model, X_test, y_test)
        test_results.append(metrics)
        print(
            f"  {name}: accuracy={metrics['Accuracy'] * 100:.2f}% | "
            f"precision={metrics['Precision']:.3f} | recall={metrics['Recall']:.3f} | "
            f"F1={metrics['F1 Score']:.3f} | ROC-AUC={metrics['ROC AUC']:.3f}"
        )
        probabilities = positive_class_probability(model, X_test)
        if probabilities is not None:
            fpr, tpr, auc_value = roc_curve(y_test == "fake", probabilities)
            roc_data[name] = (fpr, tpr, float(auc_value))

    results_df = pd.DataFrame(test_results)
    save_model_comparison(results_df, roc_data)

    print("\nSTEP 5/6 — Selecting the best model using validation F1-score")
    best_name = max(validation_scores, key=validation_scores.get)
    best_model = fitted_models[best_name]
    bundle = {
        "model": best_model,
        "best_model_name": best_name,
        "image_size": IMAGE_SIZE,
        "hog_settings": HOG_SETTINGS,
        "feature_method": "HOG",
        "labels": ["real", "fake"],
        "validation_f1_fake": validation_scores[best_name],
    }
    joblib.dump(bundle, MODELS_DIR / "best_model.joblib")

    with (REPORTS_DIR / "validation_scores.json").open("w", encoding="utf-8") as file:
        json.dump(validation_scores, file, indent=2)
    with (REPORTS_DIR / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump({**summary, "best_model": best_name, "split": {"train": 0.6, "validation": 0.2, "test": 0.2}}, file, indent=2)

    print("\nSTEP 6/6 — Saving charts, reports and trained model")
    print("\n" + "=" * 58)
    print("             TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 58)
    print(f"Best model (validation F1): {best_name}")
    print(f"Model saved: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Reports saved in: {REPORTS_DIR}")
    print("\nGenerated outputs:")
    for filename in [
        "dataset_summary.json",
        "dataset_manifest.csv",
        "class_distribution.png",
        "sample_images.png",
        "model_comparison.csv",
        "model_comparison.png",
        "roc_curves.png",
        "logistic_regression_confusion_matrix.png",
        "random_forest_confusion_matrix.png",
        "svm_confusion_matrix.png",
    ]:
        if (REPORTS_DIR / filename).exists():
            print(f"  reports/{filename}")
    print("  models/best_model.joblib")
    print("\nNext: run the app with the Python environment used for training.")


if __name__ == "__main__":
    main()