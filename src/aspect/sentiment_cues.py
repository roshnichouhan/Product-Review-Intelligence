import re


# ============================================================
# POSITIVE SENTIMENT WORDS
# ============================================================

POSITIVE_WORDS = {
    "excellent",
    "amazing",
    "awesome",
    "great",
    "good",
    "best",
    "perfect",
    "fast",
    "beautiful",
    "premium",
    "easy",
    "useful",
    "smooth",
    "strong",
    "durable",
    "reliable",
    "happy",
    "love",
    "loved",
    "worth",
    "comfortable",
    "clear",
    "bright",
    "quick",
    "wonderful"
}


# ============================================================
# NEGATIVE SENTIMENT WORDS
# ============================================================

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "worst",
    "terrible",
    "awful",
    "horrible",
    "slow",
    "expensive",
    "cheap",
    "broken",
    "damage",
    "damaged",
    "drain",
    "drains",
    "draining",
    "lag",
    "lagging",
    "problem",
    "problems",
    "issue",
    "issues",
    "difficult",
    "hard",
    "disappointing",
    "disappointed",
    "late",
    "delay",
    "delayed",
    "waste",
    "noisy",
    "dim",
    "weak",
    "uncomfortable"
}


# ============================================================
# NEGATION WORDS
# ============================================================

NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "isn't",
    "isnt",
    "wasn't",
    "wasnt",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt"
}


# ============================================================
# ANALYZE SENTIMENT CUES
# ============================================================

def analyze_sentiment_cues(text):

    if not isinstance(text, str):
        return {
            "positive_count": 0,
            "negative_count": 0,
            "sentiment": "Neutral"
        }

    words = re.findall(
        r"[a-zA-Z']+",
        text.lower()
    )

    positive_count = 0
    negative_count = 0

    for index, word in enumerate(words):

        previous_words = words[
            max(0, index - 3):index
        ]

        has_negation = any(
            negation in previous_words
            for negation in NEGATION_WORDS
        )

        if word in POSITIVE_WORDS:

            if has_negation:
                negative_count += 1
            else:
                positive_count += 1

        elif word in NEGATIVE_WORDS:

            if has_negation:
                positive_count += 1
            else:
                negative_count += 1

    # --------------------------------------------------------
    # Determine sentiment
    # --------------------------------------------------------

    if positive_count > negative_count:

        sentiment = "Positive"

    elif negative_count > positive_count:

        sentiment = "Negative"

    else:

        sentiment = "Neutral"

    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sentiment": sentiment
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    examples = [

        "The camera quality is excellent",

        "The battery drains very quickly",

        "The price is expensive",

        "The display is bright and beautiful",

        "The performance is not good"
    ]

    for text in examples:

        result = analyze_sentiment_cues(text)

        print("\nText:")
        print(text)

        print("Result:")
        print(result)