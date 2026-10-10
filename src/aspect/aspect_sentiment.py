from pathlib import Path
import pickle
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT CUSTOM COMPONENTS
# ============================================================

from src.models.attention import AttentionLayer

from src.aspect.extractor import (
    find_aspects,
    extract_aspect_context
)

from src.aspect.sentiment_cues import (
    analyze_sentiment_cues
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sentiment_attention_bilstm.keras"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tokenizer.pkl"
)

LABEL_ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "label_encoder.pkl"
)


# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH = 200


# ============================================================
# LOAD MODEL
# ============================================================

def load_attention_model():

    model = load_model(
        MODEL_PATH,
        custom_objects={
            "AttentionLayer": AttentionLayer
        }
    )

    return model


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:

        tokenizer = pickle.load(file)

    return tokenizer


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

def load_label_encoder():

    with open(
        LABEL_ENCODER_PATH,
        "rb"
    ) as file:

        encoder = pickle.load(file)

    return encoder


# ============================================================
# NEURAL MODEL SENTIMENT
# ============================================================

def predict_neural_sentiment(
    text,
    model,
    tokenizer,
    encoder
):

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(
        padded,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        probabilities
    )

    sentiment = encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        probabilities[predicted_index]
    )

    return sentiment, confidence


# ============================================================
# HYBRID SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(
    text,
    model,
    tokenizer,
    encoder
):

    # --------------------------------------------------------
    # Neural model prediction
    # --------------------------------------------------------

    neural_sentiment, neural_confidence = (
        predict_neural_sentiment(
            text,
            model,
            tokenizer,
            encoder
        )
    )

    # --------------------------------------------------------
    # Rule-based sentiment cues
    # --------------------------------------------------------

    cue_result = analyze_sentiment_cues(
        text
    )

    positive_cues = cue_result[
        "positive_count"
    ]

    negative_cues = cue_result[
        "negative_count"
    ]

    cue_sentiment = cue_result[
        "sentiment"
    ]

    # --------------------------------------------------------
    # Hybrid decision
    # --------------------------------------------------------
    # If clear positive/negative cues exist,
    # use cue sentiment for local aspect context.
    #
    # Otherwise, fall back to neural model.
    # --------------------------------------------------------

    if positive_cues > 0 or negative_cues > 0:

        sentiment = cue_sentiment

        # Simple confidence based on cue strength.
        total_cues = (
            positive_cues
            + negative_cues
        )

        confidence = (
            max(
                positive_cues,
                negative_cues
            )
            / total_cues
        )

    else:

        sentiment = neural_sentiment
        confidence = neural_confidence

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "neural_sentiment": neural_sentiment,
        "neural_confidence": neural_confidence,
        "positive_cues": positive_cues,
        "negative_cues": negative_cues
    }


# ============================================================
# ANALYZE REVIEW
# ============================================================

def analyze_review(
    review,
    model,
    tokenizer,
    encoder
):

    # --------------------------------------------------------
    # Detect aspects
    # --------------------------------------------------------

    aspects = find_aspects(
        review
    )

    # --------------------------------------------------------
    # Overall sentiment
    # --------------------------------------------------------

    overall_result = predict_sentiment(
        review,
        model,
        tokenizer,
        encoder
    )

    # --------------------------------------------------------
    # Aspect-level analysis
    # --------------------------------------------------------

    aspect_results = []

    for aspect in aspects:

        context = extract_aspect_context(
            review,
            aspect
        )

        if not context:
            continue

        sentiment_result = predict_sentiment(
            context,
            model,
            tokenizer,
            encoder
        )

        aspect_results.append({

            "aspect": aspect,

            "context": context,

            "sentiment": sentiment_result[
                "sentiment"
            ],

            "confidence": sentiment_result[
                "confidence"
            ],

            "neural_sentiment": sentiment_result[
                "neural_sentiment"
            ],

            "neural_confidence": sentiment_result[
                "neural_confidence"
            ],

            "positive_cues": sentiment_result[
                "positive_cues"
            ],

            "negative_cues": sentiment_result[
                "negative_cues"
            ]
        })

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "review": review,

        "overall_sentiment": overall_result[
            "sentiment"
        ],

        "overall_confidence": overall_result[
            "confidence"
        ],

        "overall_neural_sentiment": overall_result[
            "neural_sentiment"
        ],

        "overall_neural_confidence": overall_result[
            "neural_confidence"
        ],

        "aspects": aspect_results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 9 - ASPECT-BASED SENTIMENT ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = load_attention_model()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer()

    print("Tokenizer loaded successfully.")

    # --------------------------------------------------------
    # Load label encoder
    # --------------------------------------------------------

    print("\nLoading label encoder...")

    encoder = load_label_encoder()

    print("Label encoder loaded successfully.")

    # --------------------------------------------------------
    # Test review
    # --------------------------------------------------------

    review = (
        "The camera quality is excellent "
        "but the battery drains very quickly "
        "and the price is expensive."
    )

    print("\n" + "=" * 70)
    print("REVIEW")
    print("=" * 70)

    print(review)

    # --------------------------------------------------------
    # Analyze review
    # --------------------------------------------------------

    result = analyze_review(
        review,
        model,
        tokenizer,
        encoder
    )

    # --------------------------------------------------------
    # Overall sentiment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL SENTIMENT")
    print("=" * 70)

    print(
        "Sentiment:",
        result["overall_sentiment"]
    )

    print(
        "Confidence:",
        f"{result['overall_confidence']:.2%}"
    )

    print(
        "Neural Model:",
        result["overall_neural_sentiment"]
    )

    print(
        "Neural Confidence:",
        f"{result['overall_neural_confidence']:.2%}"
    )

    # --------------------------------------------------------
    # Aspect sentiment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ASPECT SENTIMENT")
    print("=" * 70)

    if not result["aspects"]:

        print(
            "No supported aspects detected."
        )

    else:

        for item in result["aspects"]:

            print(
                f"\nAspect: {item['aspect']}"
            )

            print(
                f"Sentiment: {item['sentiment']}"
            )

            print(
                f"Confidence: "
                f"{item['confidence']:.2%}"
            )

            print(
                f"Neural Model: "
                f"{item['neural_sentiment']}"
            )

            print(
                f"Neural Confidence: "
                f"{item['neural_confidence']:.2%}"
            )

            print(
                f"Positive Cues: "
                f"{item['positive_cues']}"
            )

            print(
                f"Negative Cues: "
                f"{item['negative_cues']}"
            )

            print(
                f"Context: "
                f"{item['context']}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()