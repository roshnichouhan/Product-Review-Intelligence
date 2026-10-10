from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reviews_processed.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline_report.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("PRODUCT REVIEW INTELLIGENCE SYSTEM")
    print("PHASE 4 — TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    print("\nLoading dataset...")
    print(INPUT_PATH)

    if not INPUT_PATH.exists():
        print("\nERROR: Processed dataset not found!")
        print("Expected file:")
        print(INPUT_PATH)
        return None

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset loaded successfully!")

    print("\nDataset shape:")
    print(df.shape)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 70)
    print("1. PREPARING DATA")
    print("=" * 70)

    # Remove rows where text or sentiment is missing
    df = df.dropna(
        subset=["clean_text", "sentiment"]
    ).copy()

    # Remove empty text
    df = df[
        df["clean_text"].str.strip().str.len() > 0
    ].copy()

    X = df["clean_text"]
    y = df["sentiment"]

    print("\nNumber of samples:", len(df))

    print("\nSentiment distribution:")
    print(y.value_counts())

    return X, y


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n" + "=" * 70)
    print("2. TRAIN TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test


# ============================================================
# TF-IDF
# ============================================================

def create_tfidf():

    print("\n" + "=" * 70)
    print("3. TF-IDF VECTORIZATION")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    print("\nTF-IDF vectorizer created.")

    return vectorizer


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

def train_model(X_train_tfidf, y_train):

    print("\n" + "=" * 70)
    print("4. TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    print("\nTraining model...")

    model.fit(
        X_train_tfidf,
        y_train
    )

    print("Model training completed!")

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test_tfidf,
    y_test
):

    print("\n" + "=" * 70)
    print("5. MODEL EVALUATION")
    print("=" * 70)

    y_pred = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        y_pred
    )

    print(report)

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    return accuracy, report, cm


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    accuracy,
    report,
    cm
):

    print("\n" + "=" * 70)
    print("6. SAVING BASELINE REPORT")
    print("=" * 70)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PRODUCT REVIEW INTELLIGENCE SYSTEM\n"
        )

        file.write(
            "PHASE 4 — TF-IDF + LOGISTIC REGRESSION\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.4f}\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(report)

        file.write(
            "\n\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

    print("\nReport saved successfully!")

    print(
        REPORT_PATH.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    if df is None:
        return

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    vectorizer = create_tfidf()

    print("\nFitting TF-IDF on training data...")

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    print(
        "Training TF-IDF shape:",
        X_train_tfidf.shape
    )

    print("\nTransforming test data...")

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        "Testing TF-IDF shape:",
        X_test_tfidf.shape
    )

    model = train_model(
        X_train_tfidf,
        y_train
    )

    accuracy, report, cm = evaluate_model(
        model,
        X_test_tfidf,
        y_test
    )

    save_report(
        accuracy,
        report,
        cm
    )

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()