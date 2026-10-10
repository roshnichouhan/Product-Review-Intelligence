from pathlib import Path

import pandas as pd

from text_cleaner import clean_text, create_sentiment


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eda_reviews.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reviews_processed.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    print("=" * 70)
    print("PRODUCT REVIEW INTELLIGENCE SYSTEM")
    print("PHASE 3 — TEXT PREPROCESSING")
    print("=" * 70)

    print("\nInput file:")
    print(INPUT_PATH)

    if not INPUT_PATH.exists():

        print("\nERROR: Input dataset not found.")
        print("Expected:")
        print(INPUT_PATH)

        return None

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset loaded successfully!")

    print("\nShape:")
    print(df.shape)

    return df


# ---------------------------------------------------------
# HANDLE MISSING TEXT
# ---------------------------------------------------------

def handle_missing_text(df):

    print("\n" + "=" * 70)
    print("1. HANDLING MISSING TEXT")
    print("=" * 70)

    before = len(df)

    df["text"] = df["text"].fillna("")

    df["title"] = df["title"].fillna("")

    print("Missing text values before cleaning:",
          df["text"].isna().sum())

    print("Rows before:",
          before)

    return df


# ---------------------------------------------------------
# COMBINE TITLE + REVIEW
# ---------------------------------------------------------

def combine_text(df):

    print("\n" + "=" * 70)
    print("2. COMBINING TITLE AND REVIEW")
    print("=" * 70)

    df["full_text"] = (
        df["title"].astype(str).str.strip()
        + " "
        + df["text"].astype(str).str.strip()
    )

    df["full_text"] = (
        df["full_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("Created column: full_text")

    return df


# ---------------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------------

def clean_reviews(df):

    print("\n" + "=" * 70)
    print("3. CLEANING REVIEW TEXT")
    print("=" * 70)

    print("Cleaning reviews...")

    df["clean_text"] = df["full_text"].apply(clean_text)

    print("Text cleaning completed.")

    return df


# ---------------------------------------------------------
# REMOVE EMPTY REVIEWS
# ---------------------------------------------------------

def remove_empty_reviews(df):

    print("\n" + "=" * 70)
    print("4. REMOVING EMPTY REVIEWS")
    print("=" * 70)

    before = len(df)

    df = df[
        df["clean_text"].str.strip().str.len() > 0
    ].copy()

    removed = before - len(df)

    print("Rows before:", before)
    print("Empty reviews removed:", removed)
    print("Rows after:", len(df))

    return df


# ---------------------------------------------------------
# CREATE SENTIMENT LABEL
# ---------------------------------------------------------

def create_labels(df):

    print("\n" + "=" * 70)
    print("5. CREATING SENTIMENT LABELS")
    print("=" * 70)

    df["sentiment"] = (
        df["rating"]
        .apply(create_sentiment)
    )

    print("\nSentiment distribution:")

    print(
        df["sentiment"]
        .value_counts()
    )

    print("\nSentiment percentage:")

    print(
        (
            df["sentiment"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    return df


# ---------------------------------------------------------
# CREATE TEXT STATISTICS
# ---------------------------------------------------------

def create_text_features(df):

    print("\n" + "=" * 70)
    print("6. CREATING TEXT FEATURES")
    print("=" * 70)

    df["char_count"] = (
        df["clean_text"]
        .str.len()
    )

    df["word_count"] = (
        df["clean_text"]
        .str.split()
        .str.len()
    )

    print("\nText statistics:")

    print(
        df[
            ["char_count", "word_count"]
        ].describe()
    )

    return df


# ---------------------------------------------------------
# SAVE DATA
# ---------------------------------------------------------

def save_data(df):

    print("\n" + "=" * 70)
    print("7. SAVING PROCESSED DATASET")
    print("=" * 70)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nProcessed dataset saved successfully!")

    print("\nLocation:")
    print(OUTPUT_PATH.resolve())

    print("\nFinal shape:")
    print(df.shape)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    df = load_data()

    if df is None:
        return

    df = handle_missing_text(df)

    df = combine_text(df)

    df = clean_reviews(df)

    df = remove_empty_reviews(df)

    df = create_labels(df)

    df = create_text_features(df)

    save_data(df)

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()