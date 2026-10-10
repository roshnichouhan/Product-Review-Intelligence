from huggingface_hub import list_repo_files


REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"


print("=" * 70)
print("TESTING HUGGING FACE CONNECTION")
print("=" * 70)

print("\nConnecting to Hugging Face...")

files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("\nConnection successful!")

print("\nNumber of files found:")
print(len(files))

print("\nFirst 30 files:")

for file in files[:30]:
    print("-", file)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)