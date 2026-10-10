from pathlib import Path
import pickle

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tokenizer.pkl"
)

SEQUENCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_WORDS = 30000
MAX_LENGTH = 200


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("PHASE 6 — TOKENIZATION")
    print("=" * 70)

    print("\nLoading processed dataset...")

    if not INPUT_PATH.exists():

        print("\nERROR: Dataset not found!")
        print(INPUT_PATH)

        return None

    df = pd.read_csv(INPUT_PATH)

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
# CREATE TOKENIZER
# ============================================================

def create_tokenizer(texts):

    print("\n" + "=" * 70)
    print("1. CREATING TOKENIZER")
    print("=" * 70)

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(texts)

    print("\nTokenizer created.")

    print(
        "Total words found:",
        len(tokenizer.word_index)
    )

    print(
        "Vocabulary used:",
        min(
            len(tokenizer.word_index),
            MAX_WORDS
        )
    )

    return tokenizer


# ============================================================
# CONVERT TEXT TO SEQUENCES
# ============================================================

def create_sequences(tokenizer, texts):

    print("\n" + "=" * 70)
    print("2. CONVERTING TEXT TO SEQUENCES")
    print("=" * 70)

    sequences = tokenizer.texts_to_sequences(
        texts
    )

    print("\nExample sequence:")

    print(sequences[0][:20])

    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    print("\nPadded sequence shape:")

    print(padded_sequences.shape)

    return padded_sequences


# ============================================================
# SAVE TOKENIZER AND SEQUENCES
# ============================================================

def save_objects(
    tokenizer,
    sequences
):

    print("\n" + "=" * 70)
    print("3. SAVING TOKENIZER")
    print("=" * 70)

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        TOKENIZER_PATH,
        "wb"
    ) as file:

        pickle.dump(
            tokenizer,
            file
        )

    print("\nTokenizer saved:")
    print(TOKENIZER_PATH.resolve())

    with open(
        SEQUENCES_PATH,
        "wb"
    ) as file:

        pickle.dump(
            sequences,
            file
        )

    print("\nSequences saved:")
    print(SEQUENCES_PATH.resolve())


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    if df is None:
        return

    texts = df["clean_text"].tolist()

    tokenizer = create_tokenizer(
        texts
    )

    sequences = create_sequences(
        tokenizer,
        texts
    )

    save_objects(
        tokenizer,
        sequences
    )

    print("\n" + "=" * 70)
    print("TOKENIZATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()