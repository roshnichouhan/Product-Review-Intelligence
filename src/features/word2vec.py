from pathlib import Path

import pandas as pd
import numpy as np

from gensim.models import Word2Vec


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reviews_processed.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "word2vec.model"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("PRODUCT REVIEW INTELLIGENCE SYSTEM")
    print("PHASE 5 — WORD2VEC & EMBEDDINGS")
    print("=" * 70)

    print("\nLoading processed reviews...")

    if not INPUT_PATH.exists():

        print("\nERROR: Dataset not found!")

        print("Expected:")
        print(INPUT_PATH)

        return None

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset loaded successfully!")

    print("Shape:", df.shape)

    return df


# ============================================================
# PREPARE SENTENCES
# ============================================================

def prepare_sentences(df):

    print("\n" + "=" * 70)
    print("1. PREPARING TEXT FOR WORD2VEC")
    print("=" * 70)

    sentences = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .apply(str.split)
        .tolist()
    )

    # Remove empty sentences
    sentences = [
        sentence
        for sentence in sentences
        if len(sentence) > 0
    ]

    print("\nNumber of sentences:", len(sentences))

    print("\nExample tokenized review:")

    print(sentences[0][:20])

    return sentences


# ============================================================
# TRAIN WORD2VEC
# ============================================================

def train_word2vec(sentences):

    print("\n" + "=" * 70)
    print("2. TRAINING WORD2VEC")
    print("=" * 70)

    print("\nTraining Word2Vec model...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=3,
        workers=4,
        sg=1,
        epochs=10,
        seed=42
    )

    print("\nWord2Vec training completed!")

    print("\nVocabulary size:")

    print(len(model.wv))

    return model


# ============================================================
# TEST WORD EMBEDDINGS
# ============================================================

def test_embeddings(model):

    print("\n" + "=" * 70)
    print("3. TESTING WORD EMBEDDINGS")
    print("=" * 70)

    words = [
        "good",
        "great",
        "battery",
        "phone",
        "camera"
    ]

    for word in words:

        if word in model.wv:

            vector = model.wv[word]

            print(
                f"\n{word}:"
            )

            print(
                "Vector shape:",
                vector.shape
            )

            print(
                "First 10 values:",
                np.round(vector[:10], 4)
            )

        else:

            print(
                f"\n'{word}' not found in vocabulary."
            )


# ============================================================
# SIMILAR WORDS
# ============================================================

def show_similar_words(model):

    print("\n" + "=" * 70)
    print("4. FINDING SIMILAR WORDS")
    print("=" * 70)

    words_to_test = [
        "good",
        "battery",
        "phone"
    ]

    for word in words_to_test:

        if word in model.wv:

            print(
                f"\nWords similar to '{word}':"
            )

            similar_words = model.wv.most_similar(
                word,
                topn=5
            )

            for similar_word, score in similar_words:

                print(
                    f"{similar_word}: {score:.4f}"
                )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    print("\n" + "=" * 70)
    print("5. SAVING WORD2VEC MODEL")
    print("=" * 70)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    print("\nWord2Vec model saved!")

    print(
        MODEL_PATH.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    if df is None:
        return

    sentences = prepare_sentences(
        df
    )

    model = train_word2vec(
        sentences
    )

    test_embeddings(
        model
    )

    show_similar_words(
        model
    )

    save_model(
        model
    )

    print("\n" + "=" * 70)
    print("PHASE 5 — WORD2VEC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()