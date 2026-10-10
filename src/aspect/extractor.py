from pathlib import Path
import sys
import re


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT ASPECTS
# ============================================================

from config.aspects import ASPECTS


# ============================================================
# FIND ASPECTS
# ============================================================

def find_aspects(text):

    if not isinstance(text, str):
        return []

    text = text.lower()

    found_aspects = []

    for aspect, keywords in ASPECTS.items():

        for keyword in keywords:

            pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, text):

                found_aspects.append(aspect)

                break

    return found_aspects


# ============================================================
# EXTRACT ASPECT CONTEXT
# ============================================================

def extract_aspect_context(text, aspect):

    if not isinstance(text, str):
        return ""

    keywords = ASPECTS.get(
        aspect,
        []
    )

    # --------------------------------------------------------
    # Split text into clauses
    # --------------------------------------------------------

    clauses = re.split(
        r"\s+(?:but|and|while|although|however|because|though)\s+",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Find the clause containing the aspect
    # --------------------------------------------------------

    for clause in clauses:

        clause_lower = clause.lower()

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword.lower())
                + r"\b"
            )

            if re.search(
                pattern,
                clause_lower
            ):

                return clause.strip()

    return ""


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    example = (
        "The camera quality is excellent "
        "but the battery drains very quickly "
        "and the price is expensive."
    )

    print("=" * 70)
    print("ASPECT EXTRACTION TEST")
    print("=" * 70)

    print("\nReview:")
    print(example)

    aspects = find_aspects(example)

    print("\nDetected aspects:")
    print(aspects)

    print("\nAspect contexts:")

    for aspect in aspects:

        context = extract_aspect_context(
            example,
            aspect
        )

        print(
            f"\n{aspect}:"
        )

        print(context)