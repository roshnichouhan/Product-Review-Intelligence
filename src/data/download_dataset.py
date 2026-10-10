from pathlib import Path
import requests
import json
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# Project root:
# Product-Review-Intelligence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset will be saved here:
# Product-Review-Intelligence/data/raw/

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "amazon_cell_phone_reviews_100k.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_REVIEWS = 100_000


# Current Hugging Face raw review file
DATA_URL = (
    "https://huggingface.co/datasets/"
    "McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/"
    "Cell_Phones_and_Accessories.jsonl"
    "?download=true"
)


# ============================================================
# DOWNLOAD REVIEWS
# ============================================================

def download_reviews():

    print("=" * 70)
    print("AMAZON REVIEWS 2023")
    print("Product Review Intelligence System")
    print("=" * 70)

    print("\nCategory:")
    print("Cell Phones and Accessories")

    print(f"\nTarget reviews: {MAX_REVIEWS:,}")

    print("\nConnecting to Hugging Face...")

    response = requests.get(
        DATA_URL,
        stream=True,
        timeout=120,
        headers={
            "User-Agent": "Product-Review-Intelligence/1.0"
        }
    )

    response.raise_for_status()

    print("Connection successful!")

    print("\nReading reviews...")
    print("The program will stop after 100,000 reviews.")

    reviews = []

    for line in response.iter_lines(
        decode_unicode=True
    ):

        if not line:
            continue

        try:

            review = json.loads(line)

            reviews.append(review)

        except json.JSONDecodeError:

            continue

        # Progress message
        if len(reviews) % 10_000 == 0:

            print(
                f"Reviews collected: "
                f"{len(reviews):,}"
            )

        # Stop after required number
        if len(reviews) >= MAX_REVIEWS:

            break

    # Close connection
    response.close()

    print("\nFinished reading reviews.")

    return reviews


# ============================================================
# PROCESS DATA
# ============================================================

def process_data(reviews):

    print("\nConverting reviews to DataFrame...")

    df = pd.DataFrame(reviews)

    print("\nOriginal shape:")
    print(df.shape)

    print("\nAvailable columns:")

    for column in df.columns:

        print("-", column)

    # --------------------------------------------------------
    # Select useful columns
    # --------------------------------------------------------

    useful_columns = [
        "rating",
        "title",
        "text",
        "timestamp",
        "verified_purchase",
        "helpful_vote",
        "asin",
        "parent_asin"
    ]

    existing_columns = [
        column
        for column in useful_columns
        if column in df.columns
    ]

    df = df[existing_columns]

    # --------------------------------------------------------
    # Clean review text
    # --------------------------------------------------------

    if "text" in df.columns:

        df["text"] = (
            df["text"]
            .fillna("")
            .astype(str)
        )

        df = df[
            df["text"].str.strip().str.len() > 0
        ]

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    print(
        f"\nDuplicate rows removed: {removed}"
    )

    print("\nFinal shape:")
    print(df.shape)

    return df


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):

    print("\nSaving dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset saved successfully!")

    print("\nExact location:")

    print(
        OUTPUT_FILE.resolve()
    )

    # File size
    size_mb = (
        OUTPUT_FILE.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"\nCSV size: {size_mb:.2f} MB"
    )


# ============================================================
# SHOW SUMMARY
# ============================================================

def show_summary(df):

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print("\nShape:")

    print(df.shape)

    print("\nColumns:")

    for column in df.columns:

        print("-", column)

    # --------------------------------------------------------
    # Rating distribution
    # --------------------------------------------------------

    if "rating" in df.columns:

        print("\nRating distribution:")

        print(
            df["rating"]
            .value_counts()
            .sort_index()
        )

    # --------------------------------------------------------
    # Verified purchases
    # --------------------------------------------------------

    if "verified_purchase" in df.columns:

        print("\nVerified purchase:")

        print(
            df["verified_purchase"]
            .value_counts()
        )

    # --------------------------------------------------------
    # Helpful votes
    # --------------------------------------------------------

    if "helpful_vote" in df.columns:

        print("\nHelpful vote statistics:")

        print(
            df["helpful_vote"].describe()
        )

    # --------------------------------------------------------
    # First reviews
    # --------------------------------------------------------

    print("\nFirst 5 reviews:")

    display_columns = [
        column
        for column in [
            "rating",
            "title",
            "text"
        ]
        if column in df.columns
    ]

    print(
        df[display_columns]
        .head()
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        reviews = download_reviews()

        if not reviews:

            print(
                "\nNo reviews were downloaded."
            )

            return

        df = process_data(reviews)

        save_dataset(df)

        show_summary(df)

        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)

        print(
            "\nYour real-world dataset is ready."
        )

        print(
            "\nNext folder:"
        )

        print(
            "data/raw/"
        )

    except requests.exceptions.RequestException as e:

        print("\n" + "=" * 70)
        print("DOWNLOAD ERROR")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()