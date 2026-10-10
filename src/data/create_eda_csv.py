from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon_cell_phone_reviews_100k.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eda_reviews.csv"
)


def main():

    print("=" * 70)
    print("CREATING EDA CSV")
    print("=" * 70)

    print("\nInput:")
    print(INPUT_PATH)

    print("\nOutput:")
    print(OUTPUT_PATH)

    # Check input
    if not INPUT_PATH.exists():
        print("\nERROR: Raw dataset not found!")
        return

    # Create processed folder
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load raw dataset
    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully!")

    print("\nOriginal shape:")
    print(df.shape)

    # Create review length
    df["review_length"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    # Save
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nEDA CSV created successfully!")

    print("\nFinal shape:")
    print(df.shape)

    print("\nSaved at:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()