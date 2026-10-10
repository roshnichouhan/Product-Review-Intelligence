import re
import pandas as pd


def clean_text(text):
    """
    Clean a product review for NLP processing.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert HTML line breaks to spaces
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters, numbers, and basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_sentiment(rating):
    """
    Convert numerical rating into a sentiment class.
    """

    if rating in [1, 2]:
        return "Negative"

    elif rating == 3:
        return "Neutral"

    elif rating in [4, 5]:
        return "Positive"

    return None