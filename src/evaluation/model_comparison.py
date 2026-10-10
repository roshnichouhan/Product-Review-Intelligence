
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import load_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_processed.csv"
SEQUENCES_PATH = PROJECT_ROOT / "data" / "processed" / "sequences.pkl"
BILSTM_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_bilstm.keras"
ATTENTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "sentiment_attention_bilstm.keras"
)
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.txt"
CSV_REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

BATCH_SIZE = 128
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("PHASE 8 - MODEL EVALUATION AND COMPARISON")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The processed dataset is empty.")

    required_columns = {"sentiment", "clean_text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["sentiment"].isna().any():
        raise ValueError("Sentiment column contains missing labels.")

    print("Dataset shape:", df.shape)
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    return df


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences():
    print("\nLoading sequences...")

    if not SEQUENCES_PATH.exists():
        raise FileNotFoundError(
            f"Sequences file not found: {SEQUENCES_PATH}"
        )

    with open(SEQUENCES_PATH, "rb") as file:
        sequences = pickle.load(file)

    sequences = np.asarray(sequences)

    if sequences.ndim != 2:
        raise ValueError(
            f"Expected 2D sequences, got shape {sequences.shape}"
        )

    print("Sequences shape:", sequences.shape)
    return sequences


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

def load_label_encoder():
    print("\nLoading label encoder...")

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {LABEL_ENCODER_PATH}"
        )

    with open(LABEL_ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    print("Classes:", list(encoder.classes_))
    return encoder


# ============================================================
# CREATE SHARED TRAIN-TEST SPLIT
# ============================================================

def create_test_split(sequences, df, encoder):
    print("\nCreating shared train-test split...")

    if len(sequences) != len(df):
        raise ValueError(
            f"Row mismatch: {len(sequences)} sequences, "
            f"{len(df)} dataset rows."
        )

    y = encoder.transform(df["sentiment"].astype(str))
    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training rows:", len(train_indices))
    print("Testing rows:", len(test_indices))

    return train_indices, test_indices, y[train_indices], y[test_indices]


# ============================================================
# EVALUATE ANY CLASSIFIER
# ============================================================

def calculate_metrics(y_test, predictions, encoder, model_name):
    labels = np.arange(len(encoder.classes_))

    accuracy = accuracy_score(y_test, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=encoder.classes_,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print(f"\nModel: {model_name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
    }


# ============================================================
# TF-IDF + LOGISTIC REGRESSION BASELINE
# ============================================================

def evaluate_tfidf_baseline(
    df, train_indices, test_indices, y_train, y_test, encoder
):
    print("\n" + "=" * 70)
    print("EVALUATING: TF-IDF + Logistic Regression")
    print("=" * 70)

    X_train = (
        df.iloc[train_indices]["clean_text"]
        .fillna("")
        .astype(str)
    )

    X_test = (
        df.iloc[test_indices]["clean_text"]
        .fillna("")
        .astype(str)
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("Training baseline model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return calculate_metrics(
        y_test,
        predictions,
        encoder,
        "TF-IDF + Logistic Regression",
    )


# ============================================================
# LOAD NEURAL MODELS
# ============================================================

def load_bilstm():
    print("\nLoading BiLSTM model...")

    if not BILSTM_MODEL_PATH.exists():
        print("WARNING: BiLSTM model not found.")
        return None

    try:
        model = load_model(BILSTM_MODEL_PATH)
        print("BiLSTM loaded successfully.")
        return model
    except Exception as error:
        print("ERROR loading BiLSTM:", error)
        return None


def load_attention():
    print("\nLoading BiLSTM + Attention model...")

    if not ATTENTION_MODEL_PATH.exists():
        print("WARNING: Attention model not found.")
        return None

    try:
        from src.models.attention import AttentionLayer

        model = load_model(
            ATTENTION_MODEL_PATH,
            custom_objects={"AttentionLayer": AttentionLayer},
        )

        print("Attention model loaded successfully.")
        return model

    except Exception as error:
        print("ERROR loading Attention model:", error)
        return None


# ============================================================
# EVALUATE NEURAL MODEL
# ============================================================

def evaluate_neural_model(model, X_test, y_test, encoder, model_name):
    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    predictions = np.argmax(probabilities, axis=1)

    return calculate_metrics(
        y_test,
        predictions,
        encoder,
        model_name,
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(result, encoder):
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"],
        display_labels=encoder.classes_,
    )

    display.plot(xticks_rotation=45, cmap="Blues")
    plt.title(f"Confusion Matrix - {result['model']}")
    plt.tight_layout()

    filename = (
        result["model"]
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )

    output_path = FIGURES_PATH / f"{filename}_confusion_matrix.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix saved:", output_path)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

def create_comparison_table(results):
    rows = [
        {
            "Model": result["model"],
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-Score": result["f1"],
        }
        for result in results
    ]

    comparison = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(comparison.round(4).to_string(index=False))

    return comparison


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(comparison, results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("PRODUCT REVIEW INTELLIGENCE SYSTEM\n")
        file.write("PHASE 8 - MODEL EVALUATION AND COMPARISON\n")
        file.write("=" * 70 + "\n\n")
        file.write("MODEL COMPARISON\n\n")
        file.write(comparison.to_string(index=False))
        file.write("\n\n")

        for result in results:
            file.write("=" * 70 + "\n")
            file.write(result["model"] + "\n")
            file.write("=" * 70 + "\n\n")
            file.write(result["report"] + "\n")
            file.write("Confusion Matrix:\n")
            file.write(str(result["confusion_matrix"]) + "\n\n")

    comparison.to_csv(CSV_REPORT_PATH, index=False)

    print("\nText report saved:", REPORT_PATH.resolve())
    print("CSV report saved:", CSV_REPORT_PATH.resolve())


# ============================================================
# MAIN
# ============================================================

def main():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    df = load_data()
    sequences = load_sequences()
    encoder = load_label_encoder()

    train_indices, test_indices, y_train, y_test = create_test_split(
        sequences, df, encoder
    )

    X_test = sequences[test_indices]
    results = []

    # 1. Traditional machine-learning baseline
    baseline_result = evaluate_tfidf_baseline(
        df,
        train_indices,
        test_indices,
        y_train,
        y_test,
        encoder,
    )

    results.append(baseline_result)
    save_confusion_matrix(baseline_result, encoder)

    # 2. BiLSTM
    bilstm = load_bilstm()

    if bilstm is not None:
        result = evaluate_neural_model(
            bilstm, X_test, y_test, encoder, "BiLSTM"
        )
        results.append(result)
        save_confusion_matrix(result, encoder)

    # 3. BiLSTM + Attention
    attention_model = load_attention()

    if attention_model is not None:
        result = evaluate_neural_model(
            attention_model,
            X_test,
            y_test,
            encoder,
            "BiLSTM + Attention",
        )
        results.append(result)
        save_confusion_matrix(result, encoder)

    if not results:
        print("No models available for evaluation.")
        return

    comparison = create_comparison_table(results)
    save_reports(comparison, results)

    print("\n" + "=" * 70)
    print("PHASE 8 EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
