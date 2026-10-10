from huggingface_hub import list_repo_files


REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"


def main():

    print("=" * 70)
    print("FINDING AMAZON REVIEW FILES")
    print("=" * 70)

    print("\nConnecting to Hugging Face...")

    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset"
    )

    print("\nConnection successful!")

    print(f"\nTotal files: {len(files)}")

    print("\nSearching for Cell Phones files...\n")

    matches = []

    for file in files:

        if "Cell_Phones_and_Accessories" in file:
            matches.append(file)

    print(f"Found {len(matches)} matching files.")

    print("\n" + "-" * 70)

    for file in matches:
        print(file)

    print("-" * 70)

    print("\nSearch completed.")


if __name__ == "__main__":
    main()