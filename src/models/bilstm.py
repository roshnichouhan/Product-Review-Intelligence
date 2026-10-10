from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reviews_processed.csv"
)

SEQUENCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences.pkl"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tokenizer.pkl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sentiment_bilstm.keras"
)

LABEL_ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "label_encoder.pkl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bilstm_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_WORDS = 30000
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 10


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("PRODUCT REVIEW INTELLIGENCE SYSTEM")
    print("PHASE 6 — BiLSTM SENTIMENT MODEL")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=["clean_text", "sentiment"]
    ).copy()

    df = df[
        df["clean_text"].str.strip().str.len() > 0
    ].copy()

    print("\nDataset shape:")
    print(df.shape)

    return df


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences():

    print("\n" + "=" * 70)
    print("1. LOADING TOKENIZED SEQUENCES")
    print("=" * 70)

    with open(
        SEQUENCES_PATH,
        "rb"
    ) as file:

        sequences = pickle.load(file)

    print("\nSequence shape:")
    print(sequences.shape)

    return sequences


# ============================================================
# ENCODE LABELS
# ============================================================

def encode_labels(df):

    print("\n" + "=" * 70)
    print("2. ENCODING SENTIMENT LABELS")
    print("=" * 70)

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df["sentiment"]
    )

    print("\nClasses:")

    print(
        list(encoder.classes_)
    )

    print("\nEncoded classes:")

    print(
        np.unique(y)
    )

    return y, encoder


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_data(
    sequences,
    y
):

    print("\n" + "=" * 70)
    print("3. TRAIN TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining shape:")
    print(X_train.shape)

    print("\nTesting shape:")
    print(X_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# BUILD BiLSTM
# ============================================================

def build_model():

    print("\n" + "=" * 70)
    print("4. BUILDING BiLSTM MODEL")
    print("=" * 70)

    model = Sequential(
        [

            Embedding(
                input_dim=MAX_WORDS,
                output_dim=EMBEDDING_DIM,
                mask_zero=True
            ),

            Bidirectional(
                LSTM(
                    LSTM_UNITS,
                    return_sequences=False
                )
            ),

            Dropout(0.4),

            Dense(
                64,
                activation="relu"
            ),

            Dropout(0.3),

            Dense(
                3,
                activation="softmax"
            )

        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel created successfully!")

    print("\nModel summary:")

    model.summary()

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("5. TRAINING BiLSTM")
    print("=" * 70)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )

    print("\nTraining started...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    print("\nTraining completed!")

    return history


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    encoder
):

    print("\n" + "=" * 70)
    print("6. EVALUATING BiLSTM")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nTest Accuracy:")
    print(
        f"{accuracy:.4f}"
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )

    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, report, cm


# ============================================================
# SAVE LABEL ENCODER
# ============================================================

def save_label_encoder(
    encoder
):

    with open(
        LABEL_ENCODER_PATH,
        "wb"
    ) as file:

        pickle.dump(
            encoder,
            file
        )

    print("\nLabel encoder saved:")
    print(
        LABEL_ENCODER_PATH.resolve()
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    accuracy,
    report,
    cm
):

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
            "PHASE 6 — BiLSTM SENTIMENT MODEL\n\n"
        )

        file.write(
            f"Test Accuracy: {accuracy:.4f}\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(
            report
        )

        file.write(
            "\n\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

    print("\nReport saved:")
    print(
        REPORT_PATH.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    sequences = load_sequences()

    y, encoder = encode_labels(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        sequences,
        y
    )

    model = build_model()

    history = train_model(
        model,
        X_train,
        y_train
    )

    accuracy, report, cm = evaluate_model(
        model,
        X_test,
        y_test,
        encoder
    )

    save_label_encoder(
        encoder
    )

    save_report(
        accuracy,
        report,
        cm
    )

    print("\n" + "=" * 70)
    print("PHASE 6 — BiLSTM COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()