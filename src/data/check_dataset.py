from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon_cell_phone_reviews_100k.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_data():

    print("=" * 70)
    print("PRODUCT REVIEW INTELLIGENCE SYSTEM")
    print("REAL-WORLD DATASET CHECK")
    print("=" * 70)

    print("\nDataset location:")
    print(DATA_PATH.resolve())

    if not DATA_PATH.exists():

        print("\nERROR: Dataset file was not found.")

        print(
            "\nExpected file:"
        )

        print(DATA_PATH)

        return None

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")

    return df


# ============================================================
# DATASET OVERVIEW
# ============================================================

def dataset_overview(df):

    print("\n" + "=" * 70)
    print("1. DATASET OVERVIEW")
    print("=" * 70)

    print("\nNumber of rows:")
    print(len(df))

    print("\nNumber of columns:")
    print(len(df.columns))

    print("\nDataset shape:")
    print(df.shape)

    print("\nMemory usage:")

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 * 1024)
    )

    print(f"{memory_mb:.2f} MB")


# ============================================================
# COLUMNS
# ============================================================

def show_columns(df):

    print("\n" + "=" * 70)
    print("2. COLUMNS")
    print("=" * 70)

    for i, column in enumerate(df.columns, start=1):

        print(
            f"{i}. {column}"
        )


# ============================================================
# DATA TYPES
# ============================================================

def show_data_types(df):

    print("\n" + "=" * 70)
    print("3. DATA TYPES")
    print("=" * 70)

    print(df.dtypes)


# ============================================================
# FIRST 5 ROWS
# ============================================================

def show_sample(df):

    print("\n" + "=" * 70)
    print("4. FIRST 5 REVIEWS")
    print("=" * 70)

    display_columns = [
        column
        for column in [
            "rating",
            "title",
            "text",
            "verified_purchase",
            "helpful_vote"
        ]
        if column in df.columns
    ]

    print(
        df[display_columns]
        .head()
        .to_string(index=False)
    )


# ============================================================
# MISSING VALUES
# ============================================================

def check_missing_values(df):

    print("\n" + "=" * 70)
    print("5. MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()

    missing_percentage = (
        missing / len(df) * 100
    )

    result = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_percentage.round(2)
    })

    print(result)


# ============================================================
# DUPLICATES
# ============================================================

def check_duplicates(df):

    print("\n" + "=" * 70)
    print("6. DUPLICATE ROWS")
    print("=" * 70)

    duplicates = df.duplicated().sum()

    print(
        f"Duplicate rows: {duplicates:,}"
    )


# ============================================================
# RATING DISTRIBUTION
# ============================================================

def rating_distribution(df):

    print("\n" + "=" * 70)
    print("7. RATING DISTRIBUTION")
    print("=" * 70)

    if "rating" not in df.columns:

        print("Rating column not found.")

        return

    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
    )

    print(rating_counts)

    print("\nPercentage:")

    rating_percentage = (
        df["rating"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        rating_percentage.round(2)
    )


# ============================================================
# VERIFIED PURCHASE
# ============================================================

def verified_purchase_distribution(df):

    print("\n" + "=" * 70)
    print("8. VERIFIED PURCHASE")
    print("=" * 70)

    if "verified_purchase" not in df.columns:

        print(
            "verified_purchase column not found."
        )

        return

    print(
        df["verified_purchase"]
        .value_counts(dropna=False)
    )


# ============================================================
# HELPFUL VOTES
# ============================================================

def helpful_votes(df):

    print("\n" + "=" * 70)
    print("9. HELPFUL VOTES")
    print("=" * 70)

    if "helpful_vote" not in df.columns:

        print(
            "helpful_vote column not found."
        )

        return

    print(
        df["helpful_vote"].describe()
    )


# ============================================================
# REVIEW LENGTH
# ============================================================

def review_length(df):

    print("\n" + "=" * 70)
    print("10. REVIEW LENGTH")
    print("=" * 70)

    if "text" not in df.columns:

        print("text column not found.")

        return

    df["review_length"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print(
        df["review_length"].describe()
    )


# ============================================================
# RATING + REVIEW LENGTH
# ============================================================

def rating_review_analysis(df):

    print("\n" + "=" * 70)
    print("11. RATING VS REVIEW LENGTH")
    print("=" * 70)

    if (
        "rating" not in df.columns
        or "text" not in df.columns
    ):

        print("Required columns not found.")

        return

    df["review_length"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    result = (
        df.groupby("rating")["review_length"]
        .agg(
            ["count", "mean", "median", "min", "max"]
        )
        .round(2)
    )

    print(result)


# ============================================================
# TEXT EXAMPLES
# ============================================================

def show_text_examples(df):

    print("\n" + "=" * 70)
    print("12. REVIEW EXAMPLES")
    print("=" * 70)

    if "text" not in df.columns:

        return

    for index, text in enumerate(
        df["text"].dropna().head(5),
        start=1
    ):

        print(f"\nReview {index}:")
        print(text)


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    if df is None:

        return

    dataset_overview(df)

    show_columns(df)

    show_data_types(df)

    show_sample(df)

    check_missing_values(df)

    check_duplicates(df)

    rating_distribution(df)

    verified_purchase_distribution(df)

    helpful_votes(df)

    review_length(df)

    rating_review_analysis(df)

    show_text_examples(df)

    print("\n" + "=" * 70)
    print("EDA CHECK COMPLETED")
    print("=" * 70)

    print(
        "\nNext step: "
        "Create the EDA notebook and visualizations."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()