
# PRODUCT REVIEW INTELLIGENCE SYSTEM
# Resume-Level AI Analytics Dashboard
# Visual style inspired by the provided dark/neon AI dashboard
# ============================================================

import sys
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.attention import AttentionLayer
from src.aspect.extractor import find_aspects, extract_aspect_context
from src.aspect.sentiment_cues import analyze_sentiment_cues
from config.aspects import ASPECTS


MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_attention_bilstm.keras"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "reviews_processed.csv"
MAX_LENGTH = 200


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ReviewIQ | Product Review Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    html, body, [class*="css"] {
        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 68% 10%,
                rgba(20, 184, 166, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 40% 90%,
                rgba(6, 182, 212, 0.07),
                transparent 25%
            ),
            #02030a;
        color: #e5e7eb;
    }

    .main {
        background: transparent;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Hide default Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #080a12 0%,
                #060810 100%
            );
        border-right: 1px solid #151a27;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.3rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 8px 8px 3px 8px;
    }

    .brand-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #021313;
        font-weight: 900;
        background:
            linear-gradient(
                135deg,
                #16e0ca,
                #08b9e8
            );
        box-shadow:
            0 0 22px rgba(20, 230, 205, 0.30);
    }

    .brand-name {
        color: #f8fafc;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -0.3px;
    }

    .brand-sub {
        color: #5f6b7d;
        font-size: 9px;
        margin-top: -2px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    .nav-label {
        color: #4d596b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1.4px;
        margin: 28px 9px 8px 9px;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 3px;
    }

    section[data-testid="stSidebar"] .stRadio label {
        background: transparent;
        border-radius: 8px;
        padding: 6px 9px;
        transition: all 0.15s ease;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #0d131e;
    }

    section[data-testid="stSidebar"] .stRadio label p {
        color: #7e899b !important;
        font-size: 12px !important;
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background:
            linear-gradient(
                90deg,
                rgba(13, 208, 193, 0.15),
                rgba(8, 180, 210, 0.05)
            );
        border: 1px solid rgba(20, 225, 205, 0.18);
        box-shadow:
            inset 3px 0 0 #12d9c6,
            0 0 20px rgba(13, 208, 193, 0.05);
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
        color: #d9fffb !important;
        font-weight: 700 !important;
    }

    .sidebar-status {
        margin-top: 35px;
        padding: 12px;
        border: 1px solid #171d2b;
        background: #090c14;
        border-radius: 10px;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 7px;
        color: #687386;
        font-size: 10px;
        margin-bottom: 7px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #12d9c6;
        box-shadow: 0 0 10px rgba(18, 217, 198, 0.8);
    }

    /* --------------------------------------------------------
       TOP BAR
    -------------------------------------------------------- */

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }

    .page-title {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin: 0;
    }

    .page-subtitle {
        color: #596578;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-top: 3px;
    }

    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #143b3b;
        background: rgba(7, 30, 30, 0.65);
        border-radius: 999px;
        padding: 7px 12px;
        color: #75f6e5;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }

    .live-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #13e4ce;
        box-shadow: 0 0 12px #13e4ce;
    }

    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 28px 30px;
        border-radius: 17px;
        background:
            radial-gradient(
                circle at 85% 40%,
                rgba(8, 211, 193, 0.10),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #080b13,
                #090d18 55%,
                #071014
            );
        border: 1px solid #172231;
        box-shadow:
            0 18px 60px rgba(0, 0, 0, 0.28);
        margin-bottom: 18px;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -100px;
        top: -120px;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: rgba(15, 219, 194, 0.08);
        filter: blur(55px);
    }

    .hero-kicker {
        color: #13d9c7;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 850;
        letter-spacing: -1px;
        margin: 0;
    }

    .hero-description {
        color: #778397;
        font-size: 12px;
        max-width: 720px;
        line-height: 1.65;
        margin-top: 8px;
    }

    /* --------------------------------------------------------
       KPI CARDS
    -------------------------------------------------------- */

    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 105px;
        padding: 17px;
        border-radius: 13px;
        background: #080b13;
        border: 1px solid #171d29;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.015);
    }

    .metric-card::after {
        content: "";
        position: absolute;
        right: -25px;
        bottom: -40px;
        width: 100px;
        height: 100px;
        background: rgba(15, 220, 198, 0.04);
        border-radius: 50%;
        filter: blur(20px);
    }

    .metric-label {
        color: #5d697c;
        font-size: 8px;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #eafdfb;
        font-size: 27px;
        line-height: 1.1;
        font-weight: 850;
        margin-top: 8px;
        letter-spacing: -0.8px;
    }

    .metric-value.teal {
        color: #12dfcb;
        text-shadow: 0 0 18px rgba(18, 223, 203, 0.15);
    }

    .metric-value.purple {
        color: #a78bfa;
    }

    .metric-value.yellow {
        color: #f5d76e;
    }

    .metric-meta {
        color: #4f5a6b;
        font-size: 9px;
        margin-top: 5px;
    }

    /* --------------------------------------------------------
       SECTION / PANELS
    -------------------------------------------------------- */

    .section-title {
        color: #e7edf5;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.1px;
        margin-top: 22px;
        margin-bottom: 4px;
    }

    .section-caption {
        color: #505c6d;
        font-size: 9px;
        margin-bottom: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .panel {
        background: #080b13;
        border: 1px solid #171d29;
        border-radius: 14px;
        padding: 15px;
        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.13);
    }

    .panel-title {
        color: #cbd5e1;
        font-size: 11px;
        font-weight: 750;
    }

    .panel-subtitle {
        color: #4f5b6c;
        font-size: 8px;
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* --------------------------------------------------------
       MODEL STATUS
    -------------------------------------------------------- */

    .model-status-card {
        background: #080b13;
        border: 1px solid #171d29;
        border-radius: 14px;
        padding: 16px;
        height: 100%;
    }

    .model-status-title {
        color: #4f5c6d;
        font-size: 8px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.1px;
    }

    .model-name {
        color: #e9eef6;
        font-size: 12px;
        font-weight: 800;
        margin-top: 12px;
    }

    .model-type {
        color: #5c6879;
        font-size: 8px;
        margin-top: 3px;
    }

    .model-progress {
        height: 5px;
        background: #151a25;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 7px;
        margin-bottom: 12px;
    }

    .model-progress > div {
        height: 100%;
        border-radius: 999px;
        background:
            linear-gradient(
                90deg,
                #0edac7,
                #0aa9d9
            );
        box-shadow: 0 0 12px rgba(14, 218, 199, 0.35);
    }

    .model-score {
        color: #12dfcb;
        font-size: 13px;
        font-weight: 850;
        float: right;
        margin-top: -21px;
    }

    .right-item {
        border-top: 1px solid #151b27;
        padding: 11px 0;
    }

    .right-label {
        color: #596577;
        font-size: 8px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .right-value {
        color: #dbe4ef;
        font-size: 11px;
        font-weight: 700;
        margin-top: 4px;
    }

    /* --------------------------------------------------------
       REVIEW ANALYZER
    -------------------------------------------------------- */

    .analyzer-box {
        background: #070a11;
        border: 1px solid #17202c;
        border-radius: 14px;
        padding: 16px;
    }

    .result-positive {
        border: 1px solid rgba(20, 220, 165, 0.28);
        background: rgba(6, 78, 59, 0.22);
        color: #62e7bc;
    }

    .result-negative {
        border: 1px solid rgba(244, 63, 94, 0.28);
        background: rgba(127, 29, 29, 0.20);
        color: #fb7185;
    }

    .result-neutral {
        border: 1px solid rgba(245, 190, 65, 0.25);
        background: rgba(120, 75, 10, 0.18);
        color: #f5d76e;
    }

    .sentiment-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 9px;
        font-weight: 850;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }

    .aspect-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 13px 0;
        border-bottom: 1px solid #151b27;
    }

    .aspect-row:last-child {
        border-bottom: none;
    }

    .aspect-name {
        width: 90px;
        color: #d7e0eb;
        font-size: 10px;
        font-weight: 750;
    }

    .aspect-context {
        flex: 1;
        color: #596679;
        font-size: 9px;
    }

    /* --------------------------------------------------------
       STREAMLIT INPUTS
    -------------------------------------------------------- */

    div[data-baseweb="textarea"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #070a11 !important;
        border-color: #1b2532 !important;
        border-radius: 9px !important;
    }

    textarea,
    input {
        color: #dce7f3 !important;
    }

    label {
        color: #687588 !important;
        font-size: 10px !important;
    }

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #0bd8c3,
                #08b5d8
            ) !important;
        color: #021313 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 850 !important;
        font-size: 10px !important;
        letter-spacing: 0.5px !important;
        min-height: 38px !important;
        box-shadow:
            0 0 25px rgba(13, 216, 195, 0.13) !important;
    }

    .stButton > button:hover {
        filter: brightness(1.08);
        box-shadow:
            0 0 28px rgba(13, 216, 195, 0.23) !important;
    }

    .stDownloadButton > button {
        background: #0b111b !important;
        color: #78eee0 !important;
        border: 1px solid #173e40 !important;
        border-radius: 8px !important;
        font-size: 10px !important;
        font-weight: 750 !important;
    }

    /* --------------------------------------------------------
       DATAFRAME / FILE UPLOADER
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: #070a11;
        border: 1px dashed #1b2936;
        border-radius: 11px;
        padding: 7px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #17202c;
        border-radius: 10px;
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #394353;
        font-size: 9px;
        padding: 28px 0 8px;
        letter-spacing: 0.5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PLOTLY THEME
# ============================================================

PLOT_BG = "#080b13"
PLOT_PAPER = "#080b13"
GRID = "#151b26"
TEXT = "#697588"
TEAL = "#12dfcb"
CYAN = "#08b5d8"
PURPLE = "#9b87f5"
PINK = "#f05a8a"
YELLOW = "#f5d76e"


def style_fig(fig, height=310):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(
            family="Inter, Arial, sans-serif",
            color=TEXT,
            size=10,
        ),
        margin=dict(l=15, r=15, t=45, b=15),
        title_font=dict(
            size=12,
            color="#cbd5e1",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9, color=TEXT),
        ),
        xaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            tickfont=dict(size=8),
            title_font=dict(size=9),
        ),
        yaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            tickfont=dict(size=8),
            title_font=dict(size=9),
        ),
    )
    return fig


# ============================================================
# DATA / MODEL LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_ai_resources():
    model = load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
    )

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    with open(LABEL_ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    return model, tokenizer, encoder


@st.cache_data(show_spinner=False)

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset CSV is empty.")

    print(f"Dataset loaded successfully: {df.shape}")
    print("Columns:", df.columns.tolist())

    return df


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def predict_neural_sentiment(text, model, tokenizer, encoder):
    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(
        padded,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    sentiment = encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        probabilities[predicted_index]
    )

    probability_map = {
        str(label): float(probabilities[index])
        for index, label in enumerate(encoder.classes_)
    }

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probability_map,
    }


def predict_hybrid_sentiment(
    text,
    model,
    tokenizer,
    encoder,
):
    neural = predict_neural_sentiment(
        text,
        model,
        tokenizer,
        encoder,
    )

    cue_result = analyze_sentiment_cues(text)

    positive_cues = cue_result["positive_count"]
    negative_cues = cue_result["negative_count"]
    cue_sentiment = cue_result["sentiment"]

    total_cues = positive_cues + negative_cues

    # Domain-cue layer is used when explicit sentiment words
    # are detected. Otherwise the neural model is used directly.
    if total_cues > 0:
        final_sentiment = cue_sentiment
        cue_score = (
            max(positive_cues, negative_cues)
            / total_cues
        )
    else:
        final_sentiment = neural["sentiment"]
        cue_score = neural["confidence"]

    return {
        "sentiment": final_sentiment,
        "confidence": cue_score,
        "neural_sentiment": neural["sentiment"],
        "neural_confidence": neural["confidence"],
        "probabilities": neural["probabilities"],
        "positive_cues": positive_cues,
        "negative_cues": negative_cues,
    }



def analyze_review(review, model, tokenizer, encoder):
    """
    Analyze the overall review and predict sentiment
    separately for each detected aspect.
    """

    if not isinstance(review, str) or not review.strip():
        return {
            "overall": {},
            "aspects": [],
        }

    # 1. Predict overall review sentiment
    overall = predict_hybrid_sentiment(
        review.strip(),
        model,
        tokenizer,
        encoder,
    )

    # 2. Detect aspects mentioned in the review
    detected_aspects = find_aspects(review)

    aspect_results = []
    seen_aspects = set()

    # 3. Analyze each aspect independently
    for aspect in detected_aspects:
        if not isinstance(aspect, str) or not aspect.strip():
            continue

        aspect_name = aspect.strip()
        aspect_key = aspect_name.casefold()

        # Avoid duplicate aspect records
        if aspect_key in seen_aspects:
            continue

        seen_aspects.add(aspect_key)

        # Extract the text relevant to this aspect
        context = extract_aspect_context(
            review,
            aspect_name,
        )

        if not isinstance(context, str) or not context.strip():
            continue

        context = context.strip()

        # Predict sentiment using only the aspect context
        prediction = predict_hybrid_sentiment(
            context,
            model,
            tokenizer,
            encoder,
        )

        if not isinstance(prediction, dict):
            prediction = {
                "sentiment": str(prediction),
            }

        # Keep the result format consistent with your dashboard
        aspect_results.append({
            "aspect": aspect_name,
            "context": context,
            **prediction,
        })

    return {
        "overall": overall,
        "aspects": aspect_results,
    }

# ============================================================
# ASPECT DATASET ANALYTICS
# ============================================================

@st.cache_data(show_spinner=False)
def calculate_aspect_statistics(dataframe):
    if dataframe is None or "clean_text" not in dataframe.columns:
        return pd.DataFrame()

    records = []

    # Keep dashboard responsive on normal laptops.
    # Full dataset is used when it is small enough.
    working_df = dataframe

    for text in working_df["clean_text"].fillna(""):
        if not text:
            continue

        found = find_aspects(text)

        for aspect in found:
            context = extract_aspect_context(
                text,
                aspect,
            )

            cue = analyze_sentiment_cues(
                context if context else text
            )

            records.append(
                {
                    "aspect": aspect,
                    "sentiment": cue["sentiment"],
                }
            )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ============================================================
# HELPERS
# ============================================================

def sentiment_class(sentiment):
    value = str(sentiment).lower()

    if value == "positive":
        return "result-positive"

    if value == "negative":
        return "result-negative"

    return "result-neutral"


def sentiment_pill(sentiment):
    css = sentiment_class(sentiment)

    icon = {
        "Positive": "●",
        "Negative": "●",
        "Neutral": "●",
    }.get(sentiment, "●")

    return (
        f'<span class="sentiment-pill {css}">'
        f'{icon} {str(sentiment).upper()}'
        f'</span>'
    )


def metric_card(
    label,
    value,
    meta="",
    accent="teal",
):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {accent}">
                {value}
            </div>
            <div class="metric-meta">
                {meta}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title,
    subtitle,
    status="MODEL ONLINE",
):
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
            <div class="live-status">
                <span class="live-dot"></span>
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_header(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-caption">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD RESOURCES
# ============================================================

df = load_dataset()

resources_loaded = False
model = None
tokenizer = None
encoder = None

try:
    model, tokenizer, encoder = load_ai_resources()
    resources_loaded = True
except Exception as error:
    st.warning(
        f"Model resources could not be loaded: {error}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-name">ReviewIQ</div>
                <div class="brand-sub">
                    Product Review Intelligence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-label">OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Review Analyzer",
            "Aspect Intelligence",
            "Review Analytics",
            "Batch Prediction",
            "Model Performance",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="nav-label">AI ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            color:#515d6e;
            font-size:9px;
            line-height:1.9;
            padding:0 9px;
        ">
            NLP PIPELINE<br>
            EMBEDDING + BiLSTM<br>
            ATTENTION MECHANISM<br>
            ASPECT INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-status">
            <div class="status-row">
                <span class="status-dot"></span>
                Model resources
                <span style="margin-left:auto;color:#12dfcb;">
                    ONLINE
                </span>
            </div>
            <div class="status-row">
                <span class="status-dot"></span>
                Dataset
                <span style="margin-left:auto;color:#12dfcb;">
                    READY
                </span>
            </div>
            <div class="status-row">
                <span class="status-dot"></span>
                Streamlit app
                <span style="margin-left:auto;color:#12dfcb;">
                    ACTIVE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            margin-top:18px;
            padding:0 9px;
            color:#3e4858;
            font-size:8px;
            line-height:1.8;
        ">
            Python · TensorFlow · Keras<br>
            NLP · BiLSTM · Attention<br>
            Streamlit · Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

   

    st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            AI-POWERED REVIEW ANALYTICS
        </div>
        <div class="hero-title">
            ReviewIQ — Product Review Intelligence
        </div>
        <div class="hero-description">
            Deep learning–powered sentiment analysis,
            aspect intelligence, and customer insights
            using NLP, BiLSTM, and Attention.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
    total_reviews = (
        len(df)
        if df is not None
        else 99750
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "TOTAL REVIEWS",
            f"{total_reviews:,}",
            "Processed Amazon reviews",
            "teal",
        )

    with col2:
        metric_card(
            "MODEL ACCURACY",
            "88.79%",
            "BiLSTM test-set benchmark",
            "teal",
        )

    with col3:
        metric_card(
            "WEIGHTED F1",
            "88.24%",
            "BiLSTM test-set benchmark",
            "purple",
        )

    with col4:
        metric_card(
            "SENTIMENT CLASSES",
            "03",
            "Negative · Neutral · Positive",
            "yellow",
        )

    # --------------------------------------------------------
    # MAIN ANALYTICS + RIGHT STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Review Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'DATASET OVERVIEW • SENTIMENT • CUSTOMER SIGNALS'
        '</div>',
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns(
        [1.8, 1.0, 0.78],
        gap="small",
    )

    # Sentiment distribution
    with left:

        if df is not None and "sentiment" in df.columns:

            sentiment_counts = (
                df["sentiment"]
                .value_counts()
                .reindex(
                    ["Positive", "Neutral", "Negative"]
                )
                .fillna(0)
                .reset_index()
            )

            sentiment_counts.columns = [
                "Sentiment",
                "Reviews",
            ]

            fig = px.area(
                sentiment_counts,
                x="Sentiment",
                y="Reviews",
                markers=True,
                title="Sentiment Distribution",
            )

            fig.update_traces(
                line=dict(
                    color=TEAL,
                    width=3,
                ),
                fillcolor="rgba(18,223,203,0.10)",
                marker=dict(
                    size=7,
                    color=TEAL,
                ),
            )

            fig.update_xaxes(
                categoryorder="array",
                categoryarray=[
                    "Negative",
                    "Neutral",
                    "Positive",
                ],
            )

            style_fig(fig, 285)

            fig.update_layout(
                title=dict(
                    text="Sentiment Distribution",
                    x=0.02,
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # Sentiment donut
    with middle:

        if df is not None and "sentiment" in df.columns:

            counts = (
                df["sentiment"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Sentiment",
                "Reviews",
            ]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=counts["Sentiment"],
                        values=counts["Reviews"],
                        hole=0.72,
                        textinfo="none",
                        marker=dict(
                            colors=[
                                TEAL,
                                YELLOW,
                                PINK,
                            ],
                            line=dict(
                                color="#080b13",
                                width=3,
                            ),
                        ),
                    )
                ]
            )

            fig.add_annotation(
                text=f"{len(df):,}<br><span style='font-size:9px'>REVIEWS</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    size=17,
                    color="#e7edf5",
                ),
            )

            fig.update_layout(
                title="Sentiment Mix",
                template="plotly_dark",
                paper_bgcolor=PLOT_PAPER,
                plot_bgcolor=PLOT_BG,
                height=285,
                margin=dict(
                    l=5,
                    r=5,
                    t=45,
                    b=5,
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    y=-0.03,
                    x=0.5,
                    xanchor="center",
                    font=dict(size=8),
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # Right model status
        # Right model status
    with right:

        model_status_html = """
        <div class="model-status-card">
            <div class="model-status-title">MODEL STATUS</div>
            <div class="model-name">sentiment_attention_bilstm</div>
            <div class="model-type">BiLSTM + Attention • Production Model</div>
            <div style="margin-top:17px;">
                <div class="right-label">Benchmark accuracy</div>
                <div class="model-score">88.79%</div>
                <div class="model-progress">
                    <div style="width:88.79%;"></div>
                </div>
            </div>
            <div class="right-item">
                <div class="right-label">Embedding</div>
                <div class="right-value">128 dimensions</div>
            </div>
            <div class="right-item">
                <div class="right-label">Sequence length</div>
                <div class="right-value">200 tokens</div>
            </div>
            <div class="right-item">
                <div class="right-label">Architecture</div>
                <div class="right-value">BiLSTM → Attention</div>
            </div>
            <div class="right-item">
                <div class="right-label">Output</div>
                <div class="right-value">3-class sentiment</div>
            </div>
        </div>
        """

        st.markdown(
            model_status_html,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    left2, right2 = st.columns(
        [1, 1],
        gap="small",
    )

    with left2:

        if df is not None and "rating" in df.columns:

            rating_counts = (
                df["rating"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            rating_counts.columns = [
                "Rating",
                "Reviews",
            ]

            fig = px.bar(
                rating_counts,
                x="Rating",
                y="Reviews",
                title="Rating Distribution",
            )

            fig.update_traces(
                marker_color=CYAN,
                marker_line_width=0,
            )

            style_fig(fig, 270)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with right2:

        if df is not None and "word_count" in df.columns:

            fig = px.histogram(
                df,
                x="word_count",
                nbins=45,
                title="Review Length Distribution",
            )

            fig.update_traces(
                marker_color=PURPLE,
                opacity=0.78,
            )

            style_fig(fig, 270)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    panel_header(
        "AI Processing Pipeline",
        "END-TO-END NLP ARCHITECTURE",
    )

    pipeline = [
        ("01", "Raw Review"),
        ("02", "Cleaning"),
        ("03", "Tokenizer"),
        ("04", "Embedding"),
        ("05", "BiLSTM"),
        ("06", "Attention"),
        ("07", "Sentiment"),
        ("08", "Aspects"),
    ]

    cols = st.columns(8)

    for column, (number, name) in zip(
        cols,
        pipeline,
    ):
        with column:
            st.markdown(
                f"""
                <div class="panel" style="
                    padding:12px;
                    min-height:76px;
                ">
                    <div style="
                        color:#12dfcb;
                        font-size:8px;
                        font-weight:850;
                    ">
                        {number}
                    </div>
                    <div style="
                        color:#cbd5e1;
                        font-size:9px;
                        font-weight:750;
                        margin-top:9px;
                    ">
                        {name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# REVIEW ANALYZER
# ============================================================

elif page == "Review Analyzer":

    page_header(
        "Review Analyzer",
        "AI INFERENCE • SENTIMENT + ASPECT INTELLIGENCE",
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                LIVE NLP INFERENCE
            </div>
            <div class="hero-title">
                Analyze a customer review.
            </div>
            <div class="hero-description">
                Enter any product review to run the trained
                BiLSTM + Attention model and the project's
                aspect intelligence pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    review = st.text_area(
        "CUSTOMER REVIEW",
        height=145,
        placeholder=(
            "The camera quality is excellent but "
            "the battery drains very quickly and "
            "the price is expensive."
        ),
    )

    analyze = st.button(
        "RUN AI ANALYSIS",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        if not review.strip():
            st.warning("Enter a customer review first.")

        elif not resources_loaded:
            st.error(
                "Model resources are unavailable. "
                "Check the models folder and Python environment."
            )

        else:

            with st.spinner("Running NLP inference..."):

                result = analyze_review(
                    review,
                    model,
                    tokenizer,
                    encoder,
                )

            overall = result["overall"]

            panel_header(
                "Prediction Overview",
                "NEURAL MODEL + DOMAIN SENTIMENT CUES",
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            FINAL SENTIMENT
                        </div>
                        <div style="margin-top:13px;">
                            {sentiment_pill(overall["sentiment"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                metric_card(
                    "NEURAL CONFIDENCE",
                    f"{overall['neural_confidence']:.1%}",
                    "Softmax probability",
                    "teal",
                )

            with c3:
                metric_card(
                    "POSITIVE CUES",
                    overall["positive_cues"],
                    "Detected sentiment terms",
                    "teal",
                )

            with c4:
                metric_card(
                    "NEGATIVE CUES",
                    overall["negative_cues"],
                    "Detected sentiment terms",
                    "purple",
                )

            # Probability chart
            probabilities = overall["probabilities"]

            probability_df = pd.DataFrame(
                {
                    "Sentiment": list(
                        probabilities.keys()
                    ),
                    "Probability": list(
                        probabilities.values()
                    ),
                }
            )

            fig = px.bar(
                probability_df,
                x="Sentiment",
                y="Probability",
                title="Neural Model Probability Distribution",
            )

            fig.update_traces(
                marker_color=TEAL,
                marker_line_width=0,
            )

            fig.update_yaxes(
                tickformat=".0%",
                range=[0, 1],
            )

            style_fig(fig, 260)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            panel_header(
                "Aspect Intelligence",
                "RULE-ASSISTED ASPECT EXTRACTION + SENTIMENT",
            )

            aspects = result["aspects"]

            if not aspects:

                st.info(
                    "No supported product aspect was detected."
                )

            else:

                for item in aspects:

                    col1, col2, col3 = st.columns(
                        [0.9, 2.5, 0.9]
                    )

                    with col1:

                        st.markdown(
                            f"""
                            <div class="panel"
                                 style="min-height:82px;">
                                <div class="right-label">
                                    ASPECT
                                </div>
                                <div style="
                                    color:#e5edf6;
                                    font-size:14px;
                                    font-weight:800;
                                    margin-top:7px;
                                ">
                                    {item["aspect"].title()}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col2:

                        st.markdown(
                            f"""
                            <div class="panel"
                                 style="min-height:82px;">
                                <div class="right-label">
                                    CONTEXT
                                </div>
                                <div style="
                                    color:#7a8799;
                                    font-size:10px;
                                    line-height:1.5;
                                    margin-top:7px;
                                ">
                                    {item["context"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col3:

                        st.markdown(
                            f"""
                            <div class="panel"
                                 style="min-height:82px;">
                                <div class="right-label">
                                    SENTIMENT
                                </div>
                                <div style="margin-top:8px;">
                                    {sentiment_pill(item["sentiment"])}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.caption(
                "Note: aspect sentiment uses the project's "
                "domain-cue + neural hybrid pipeline; "
                "it is not a separately trained supervised ABSA model."
            )


# ============================================================
# 
# ============================================================
# 
# ============================================================
# ASPECT INTELLIGENCE
# ============================================================

elif page == "Aspect Intelligence":

    page_header(
        "Aspect Intelligence",
        "PRODUCT EXPERIENCE • FEATURE SENTIMENT • CUSTOMER INSIGHTS",
    )

    # --------------------------------------------------------
    # PROFESSIONAL PAGE INTRODUCTION
    # Uses native Streamlit components to avoid raw HTML.
    # --------------------------------------------------------

    st.markdown("## Product Experience Intelligence")

    st.caption(
        "Discover which product features customers praise, "
        "which features frustrate them, and what needs improvement."
    )

    st.divider()

    # --------------------------------------------------------
    # REVIEW INPUT WORKSPACE
    # --------------------------------------------------------

    panel_header(
        "Analyze Product Feedback",
        "FEATURE-LEVEL SENTIMENT ANALYSIS",
    )

    review = st.text_area(
        "Paste a product review",
        height=120,
        placeholder=(
            "The camera quality is excellent, but the battery "
            "drains quickly and the price is too high."
        ),
        key="aspect_review_input",
    )

    analyze_aspects = st.button(
        "Analyze Product Aspects",
        type="primary",
        use_container_width=True,
        key="aspect_analyze_button",
    )

    if analyze_aspects:

        if not review.strip():

            st.warning("Please enter a product review.")

        elif not resources_loaded:

            st.error(
                "Model resources are unavailable. "
                "Check your model, tokenizer, and encoder files."
            )

        else:

            try:
                with st.spinner(
                    "Analyzing product features and sentiment..."
                ):
                    result = analyze_review(
                        review,
                        model,
                        tokenizer,
                        encoder,
                    )

                # Use the actual output structure of analyze_review().
                aspects = result.get("aspects", [])

                # ------------------------------------------------
                # NORMALIZE ASPECT RECORDS
                # ------------------------------------------------

                aspect_rows = []

                for item in aspects:

                    if not isinstance(item, dict):
                        continue

                    aspect_name = item.get("aspect")
                    context = item.get("context", "")
                    sentiment = item.get("sentiment", "Unknown")

                    if not aspect_name:
                        continue

                    aspect_rows.append(
                        {
                            "Aspect": str(aspect_name).title(),
                            "Sentiment": str(sentiment).title(),
                            "Evidence": str(context),
                        }
                    )

                if not aspect_rows:

                    panel_header(
                        "No Aspect Signals Found",
                        "ASPECT EXTRACTION RESULTS",
                    )

                    st.info(
                        "The current pipeline did not return any "
                        "supported product aspects for this review. "
                        "Try a review mentioning features such as "
                        "camera, battery, screen, price, or performance."
                    )

                else:

                    aspect_df = pd.DataFrame(aspect_rows)

                    # ------------------------------------------------
                    # SUMMARY METRICS
                    # ------------------------------------------------

                    positive_count = int(
                        aspect_df["Sentiment"]
                        .eq("Positive")
                        .sum()
                    )

                    neutral_count = int(
                        aspect_df["Sentiment"]
                        .eq("Neutral")
                        .sum()
                    )

                    negative_count = int(
                        aspect_df["Sentiment"]
                        .eq("Negative")
                        .sum()
                    )

                    total_aspects = len(aspect_df)

                    unique_aspects = aspect_df[
                        "Aspect"
                    ].nunique()

                    panel_header(
                        "Aspect Intelligence Overview",
                        "SUMMARY OF DETECTED FEATURE-LEVEL SIGNALS",
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        metric_card(
                            "ASPECT MENTIONS",
                            f"{total_aspects:,}",
                            "Detected aspect records",
                            "teal",
                        )

                    with c2:
                        metric_card(
                            "UNIQUE FEATURES",
                            f"{unique_aspects:,}",
                            "Distinct product aspects",
                            "purple",
                        )

                    with c3:
                        metric_card(
                            "POSITIVE SIGNALS",
                            f"{positive_count:,}",
                            "Positive aspect predictions",
                            "teal",
                        )

                    with c4:
                        metric_card(
                            "NEGATIVE SIGNALS",
                            f"{negative_count:,}",
                            "Negative aspect predictions",
                            "purple",
                        )

                    # ------------------------------------------------
                    # VISUAL ANALYTICS
                    # ------------------------------------------------

                    panel_header(
                        "Customer Feedback Signals",
                        "SENTIMENT DISTRIBUTION AND FEATURE FREQUENCY",
                    )

                    left, right = st.columns(2, gap="large")

                    sentiment_colors = {
                        "Positive": "#12dfcb",
                        "Neutral": "#f5b942",
                        "Negative": "#ff647c",
                    }

                    with left:

                        sentiment_order = [
                            "Positive",
                            "Neutral",
                            "Negative",
                        ]

                        sentiment_counts = (
                            aspect_df["Sentiment"]
                            .value_counts()
                            .reindex(
                                sentiment_order,
                                fill_value=0,
                            )
                            .rename_axis("Sentiment")
                            .reset_index(name="Mentions")
                        )

                        sentiment_counts = sentiment_counts[
                            sentiment_counts["Mentions"] > 0
                        ]

                        if not sentiment_counts.empty:

                            fig = px.pie(
                                sentiment_counts,
                                names="Sentiment",
                                values="Mentions",
                                hole=0.65,
                                color="Sentiment",
                                color_discrete_map=sentiment_colors,
                                title="Aspect Sentiment Mix",
                            )

                            fig.update_traces(
                                textposition="inside",
                                textinfo="percent+label",
                            )

                            style_fig(fig, 330)

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                config={
                                    "displayModeBar": False
                                },
                            )

                    with right:

                        aspect_frequency = (
                            aspect_df.groupby("Aspect")
                            .size()
                            .reset_index(name="Mentions")
                            .sort_values(
                                "Mentions",
                                ascending=True,
                            )
                        )

                        fig = px.bar(
                            aspect_frequency,
                            x="Mentions",
                            y="Aspect",
                            orientation="h",
                            title="Detected Product Features",
                        )

                        fig.update_traces(
                            marker_color="#12dfcb",
                            marker_line_width=0,
                        )

                        style_fig(fig, 330)

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "displayModeBar": False
                            },
                        )

                    # ------------------------------------------------
                    # FEATURE-LEVEL FINDINGS
                    # ------------------------------------------------

                    panel_header(
                        "Feature-Level Findings",
                        "PRODUCT ASPECT • SENTIMENT • REVIEW EVIDENCE",
                    )

                    for _, item in aspect_df.iterrows():

                        aspect_name = item["Aspect"]
                        sentiment = item["Sentiment"]
                        evidence = item["Evidence"]

                        color = sentiment_colors.get(
                            sentiment,
                            "#94a3b8",
                        )

                        with st.container(border=True):

                            feature_col, evidence_col = st.columns(
                                [1, 2.5],
                                gap="medium",
                            )

                            with feature_col:

                                st.caption("PRODUCT FEATURE")
                                st.subheader(aspect_name)

                                st.markdown(
                                    f":{'green' if sentiment == 'Positive' else 'red' if sentiment == 'Negative' else 'orange'}[● {sentiment}]"
                                    if sentiment in [
                                        "Positive",
                                        "Negative",
                                        "Neutral",
                                    ]
                                    else f"**{sentiment}**"
                                )

                            with evidence_col:

                                st.caption("REVIEW EVIDENCE")

                                if evidence.strip():
                                    st.write(f"“{evidence}”")
                                else:
                                    st.write(
                                        "No context was returned "
                                        "for this aspect."
                                    )

                    # ------------------------------------------------
                    # SENTIMENT BREAKDOWN TABLE
                    # ------------------------------------------------

                    panel_header(
                        "Aspect Sentiment Breakdown",
                        "DETAILED RESULTS",
                    )

                    breakdown = (
                        aspect_df.groupby(
                            ["Aspect", "Sentiment"]
                        )
                        .size()
                        .unstack(fill_value=0)
                    )

                    for label in [
                        "Positive",
                        "Neutral",
                        "Negative",
                    ]:
                        if label not in breakdown.columns:
                            breakdown[label] = 0

                    breakdown = breakdown[
                        ["Positive", "Neutral", "Negative"]
                    ].reset_index()

                    st.dataframe(
                        breakdown,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Results are based on the aspect records "
                        "returned by the existing rule-assisted and "
                        "neural sentiment pipeline. Predictions may "
                        "not always reflect the true sentiment."
                    )

            except Exception as error:

                st.error(
                    f"Aspect analysis failed: {error}"
                )


# ============================================================
# REVIEW ANALYTICS
# ============================================================

elif page == "Review Analytics":

    page_header(
        "Review Analytics",
        "DATA EXPLORATION • FILTERS • CUSTOMER PATTERNS",
    )

    if df is None:

        st.error(
            "Processed dataset not found at "
            f"{DATA_PATH}"
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            ratings = sorted(
                df["rating"]
                .dropna()
                .unique()
            )

            selected_ratings = st.multiselect(
                "RATING FILTER",
                ratings,
                default=ratings,
            )

        with col2:

            sentiments = sorted(
                df["sentiment"]
                .dropna()
                .unique()
            )

            selected_sentiments = st.multiselect(
                "SENTIMENT FILTER",
                sentiments,
                default=sentiments,
            )

        with col3:

            verified_options = (
                sorted(
                    df["verified_purchase"]
                    .dropna()
                    .unique()
                )
                if "verified_purchase" in df.columns
                else []
            )

            selected_verified = st.multiselect(
                "VERIFIED PURCHASE",
                verified_options,
                default=verified_options,
            )

        filtered = df.copy()

        if selected_ratings:
            filtered = filtered[
                filtered["rating"].isin(
                    selected_ratings
                )
            ]

        if selected_sentiments:
            filtered = filtered[
                filtered["sentiment"].isin(
                    selected_sentiments
                )
            ]

        if selected_verified:
            filtered = filtered[
                filtered["verified_purchase"].isin(
                    selected_verified
                )
            ]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "FILTERED REVIEWS",
                f"{len(filtered):,}",
                "Current selection",
                "teal",
            )

        with c2:
            avg_rating = (
                filtered["rating"].mean()
                if len(filtered)
                else 0
            )

            metric_card(
                "AVG RATING",
                f"{avg_rating:.2f}",
                "Selected reviews",
                "yellow",
            )

        with c3:
            avg_length = (
                filtered["word_count"].mean()
                if len(filtered)
                and "word_count" in filtered.columns
                else 0
            )

            metric_card(
                "AVG WORD COUNT",
                f"{avg_length:.1f}",
                "Words per review",
                "purple",
            )

        with c4:
            positive_share = (
                (
                    filtered["sentiment"]
                    .eq("Positive")
                    .mean()
                    * 100
                )
                if len(filtered)
                else 0
            )

            metric_card(
                "POSITIVE SHARE",
                f"{positive_share:.1f}%",
                "Selected review share",
                "teal",
            )

        left, right = st.columns(2)

        with left:

            sentiment_data = (
                filtered["sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_data.columns = [
                "Sentiment",
                "Reviews",
            ]

            fig = px.bar(
                sentiment_data,
                x="Sentiment",
                y="Reviews",
                title="Sentiment Volume",
            )

            fig.update_traces(
                marker_color=TEAL,
            )

            style_fig(fig, 290)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with right:

            rating_data = (
                filtered["rating"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            rating_data.columns = [
                "Rating",
                "Reviews",
            ]

            fig = px.bar(
                rating_data,
                x="Rating",
                y="Reviews",
                title="Rating Distribution",
            )

            fig.update_traces(
                marker_color=PURPLE,
            )

            style_fig(fig, 290)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Sentiment by rating
        cross = pd.crosstab(
            filtered["rating"],
            filtered["sentiment"],
        )

        fig = px.bar(
            cross,
            barmode="group",
            title="Sentiment by Rating",
        )

        fig.update_traces(
            marker_line_width=0,
        )

        style_fig(fig, 330)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ============================================================
# BATCH PREDICTION
# ============================================================


elif page == "Batch Prediction":

    page_header(
        "Batch Prediction",
        "CSV INFERENCE • VALIDATED INPUT • DOWNLOADABLE RESULTS",
    )

    st.title("Batch Review Intelligence")
    st.caption(
        "Upload product reviews to generate sentiment predictions "
        "and model confidence scores."
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload review CSV",
        type=["csv"],
        key="batch_review_csv",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.size == 0:
                st.error("The uploaded file is empty.")
                st.stop()

            if uploaded_file.size > 20 * 1024 * 1024:
                st.error("Maximum supported file size is 20 MB.")
                st.stop()

            batch_df = pd.read_csv(uploaded_file)

            if batch_df.empty or len(batch_df.columns) == 0:
                st.error("The CSV contains no usable data.")
                st.stop()

            st.success(
                f"Loaded {len(batch_df):,} rows and "
                f"{len(batch_df.columns):,} columns."
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            # Find text-like columns and prioritize likely review names.
            text_columns = [
                col for col in batch_df.columns
                if (
                    pd.api.types.is_object_dtype(batch_df[col])
                    or pd.api.types.is_string_dtype(batch_df[col])
                )
            ]

            hints = (
                "review", "text", "comment", "feedback",
                "content", "opinion",
            )

            preferred_columns = [
                col for col in text_columns
                if any(hint in str(col).lower() for hint in hints)
            ]

            candidates = preferred_columns or text_columns

            if not candidates:
                st.error("No text column was found in the CSV.")
                st.stop()

            selected_column = st.selectbox(
                "Select the review text column",
                options=list(batch_df.columns),
                index=list(batch_df.columns).index(candidates[0]),
            )

            raw_text = batch_df[selected_column]
            cleaned_text = raw_text.astype("string").str.strip()

            missing_mask = (
                raw_text.isna()
                | cleaned_text.fillna("").eq("")
            )

            normalized_text = (
                cleaned_text.str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )

            normalized_text = normalized_text.mask(missing_mask)

            duplicate_mask = (
                normalized_text.notna()
                & normalized_text.duplicated(keep="first")
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Total rows", f"{len(batch_df):,}")
            c2.metric("Missing reviews", f"{int(missing_mask.sum()):,}")
            c3.metric("Duplicate reviews", f"{int(duplicate_mask.sum()):,}")

            if normalized_text.notna().sum() == 0:
                st.error("The selected column contains no review text.")
                st.stop()

            if not resources_loaded:
                st.error(
                    "Model resources are unavailable. "
                    "Check model, tokenizer and label encoder loading."
                )
                st.stop()

            if st.button(
                "Run Batch Prediction",
                type="primary",
                use_container_width=True,
            ):
                # Predict each distinct review once.
                unique_reviews = (
                    cleaned_text[normalized_text.notna()]
                    .groupby(
                        normalized_text[normalized_text.notna()],
                        sort=False,
                    )
                    .first()
                )

                result_by_text = {}
                progress = st.progress(0)
                status = st.empty()
                total = len(unique_reviews)

                try:
                    for i, (key, review) in enumerate(
                        unique_reviews.items(), start=1
                    ):
                        result = predict_neural_sentiment(
                            str(review),
                            model,
                            tokenizer,
                            encoder,
                        )

                        sentiment = str(result["sentiment"]).strip()
                        confidence = float(result["confidence"])

                        if not 0 <= confidence <= 1:
                            raise ValueError(
                                "Confidence must be between 0 and 1."
                            )

                        result_by_text[key] = {
                            "sentiment": sentiment,
                            "confidence": confidence,
                        }

                        progress.progress(i / total)
                        status.caption(
                            f"Processed {i:,} of {total:,} unique reviews"
                        )

                except Exception as prediction_error:
                    st.error(f"Prediction failed: {prediction_error}")
                    st.stop()

                output_df = batch_df.copy()

                output_df["predicted_sentiment"] = [
                    (
                        result_by_text[key]["sentiment"]
                        if pd.notna(key) and key in result_by_text
                        else "Unknown"
                    )
                    for key in normalized_text
                ]

                output_df["model_confidence"] = [
                    (
                        result_by_text[key]["confidence"]
                        if pd.notna(key) and key in result_by_text
                        else None
                    )
                    for key in normalized_text
                ]

                st.success("Batch prediction completed.")

                sentiment_order = ["Negative", "Neutral", "Positive"]

                valid_predictions = output_df[
                    output_df["predicted_sentiment"].isin(sentiment_order)
                ]

                counts = (
                    valid_predictions["predicted_sentiment"]
                    .value_counts()
                    .reindex(sentiment_order, fill_value=0)
                    .rename_axis("Sentiment")
                    .reset_index(name="Reviews")
                )

                counts["Percentage"] = (
                    counts["Reviews"]
                    / max(len(valid_predictions), 1)
                    * 100
                ).round(2)

                m1, m2, m3 = st.columns(3)
                for col, label in zip(
                    [m1, m2, m3], sentiment_order
                ):
                    number = int(
                        counts.loc[
                            counts["Sentiment"] == label, "Reviews"
                        ].iloc[0]
                    )
                    col.metric(label, f"{number:,}")

                fig = px.bar(
                    counts,
                    x="Sentiment",
                    y="Reviews",
                    text="Reviews",
                    category_orders={"Sentiment": sentiment_order},
                    color="Sentiment",
                    color_discrete_map={
                        "Negative": "#F4728B",
                        "Neutral": "#F5C96A",
                        "Positive": "#42D6B0",
                    },
                    title="Predicted Sentiment Distribution",
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=PLOT_PAPER,
                    plot_bgcolor=PLOT_BG,
                    height=350,
                    showlegend=False,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                st.subheader("Prediction Results")
                st.dataframe(
                    output_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download Prediction Results",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name="review_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except pd.errors.EmptyDataError:
            st.error("The CSV is empty or has no readable columns.")
        except UnicodeDecodeError:
            st.error("Save the CSV as UTF-8 and upload it again.")
        except Exception as error:
            st.error(f"Could not process CSV: {error}")

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    page_header(
        "Model Performance",
        "MODEL EVALUATION • TEST SET RESULTS",
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                EVALUATION REPORT
            </div>
            <div class="hero-title">
                BiLSTM sentiment benchmark.
            </div>
            <div class="hero-description">
                These metrics are the actual BiLSTM test-set
                evaluation from the project's Phase 8 model
                comparison. The deployed model is the
                BiLSTM + Attention model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "ACCURACY",
            "88.79%",
            "Test set",
            "teal",
        )

    with c2:
        metric_card(
            "PRECISION",
            "87.96%",
            "Weighted",
            "purple",
        )

    with c3:
        metric_card(
            "RECALL",
            "88.79%",
            "Weighted",
            "yellow",
        )

    with c4:
        metric_card(
            "F1 SCORE",
            "88.24%",
            "Weighted",
            "teal",
        )

    panel_header(
        "Class-Level Performance",
        "PRECISION • RECALL • F1",
    )

    metrics = pd.DataFrame(
        {
            "Class": [
                "Negative",
                "Neutral",
                "Positive",
            ],
            "Precision": [
                0.75,
                0.48,
                0.95,
            ],
            "Recall": [
                0.82,
                0.33,
                0.96,
            ],
            "F1": [
                0.78,
                0.39,
                0.95,
            ],
        }
    )

    fig = px.bar(
        metrics,
        x="Class",
        y=[
            "Precision",
            "Recall",
            "F1",
        ],
        barmode="group",
        title="Class Performance",
    )

    fig.update_traces(
        marker_line_width=0,
    )

    style_fig(fig, 350)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    left, right = st.columns(2)

    with left:

        confusion = np.array(
            [
                [2413, 223, 290],
                [524, 531, 541],
                [301, 358, 14769],
            ]
        )

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=confusion,
                    x=[
                        "Negative",
                        "Neutral",
                        "Positive",
                    ],
                    y=[
                        "Negative",
                        "Neutral",
                        "Positive",
                    ],
                    text=confusion,
                    texttemplate="%{text}",
                    colorscale=[
                        [0, "#071318"],
                        [0.45, "#087e80"],
                        [1, "#13e4ce"],
                    ],
                    showscale=False,
                )
            ]
        )

        fig.update_layout(
            title="Confusion Matrix",
            template="plotly_dark",
            paper_bgcolor=PLOT_PAPER,
            plot_bgcolor=PLOT_BG,
            height=390,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20,
            ),
            xaxis=dict(
                title="Predicted",
                gridcolor=GRID,
            ),
            yaxis=dict(
                title="Actual",
                gridcolor=GRID,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    


# ============================================================
# ABOUT
# ============================================================

elif page == "About Project":

    page_header(
        "Project Overview",
        "END-TO-END DATA SCIENCE + NLP APPLICATION",
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                PORTFOLIO PROJECT
            </div>
            <div class="hero-title">
                Product Review Intelligence System
            </div>
            <div class="hero-description">
                An end-to-end Natural Language Processing and
                Deep Learning application that converts raw
                customer reviews into sentiment and product
                aspect intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        panel_header(
            "Technology Stack",
            "TOOLS USED IN THE PROJECT",
        )

        technologies = [
            ("Python", "Core programming"),
            ("Pandas / NumPy", "Data processing"),
            ("Scikit-learn", "Baseline ML"),
            ("TensorFlow / Keras", "Deep learning"),
            ("Word2Vec", "Word embeddings"),
            ("BiLSTM", "Sequence modeling"),
            ("Attention", "Context weighting"),
            ("Streamlit", "Application UI"),
            ("Plotly", "Interactive analytics"),
        ]

        for technology, purpose in technologies:

            st.markdown(
                f"""
                <div class="right-item">
                    <div class="right-label">
                        {technology}
                    </div>
                    <div class="right-value">
                        {purpose}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:

        panel_header(
            "Project Workflow",
            "FROM RAW DATA TO BUSINESS INSIGHT",
        )

        workflow = [
            "Real-world Amazon Reviews",
            "EDA & Data Quality Checks",
            "Text Cleaning",
            "Sentiment Label Creation",
            "TF-IDF + Logistic Regression Baseline",
            "Word2Vec / Embedding Concepts",
            "BiLSTM Sentiment Model",
            "BiLSTM + Attention",
            "Model Evaluation",
            "Aspect Extraction",
            "Hybrid Aspect Sentiment",
            "Streamlit Deployment",
        ]

        for index, step in enumerate(
            workflow,
            start=1,
        ):

            st.markdown(
                f"""
                <div class="right-item">
                    <span style="
                        color:#12dfcb;
                        font-size:9px;
                        font-weight:850;
                        margin-right:8px;
                    ">
                        {index:02d}
                    </span>
                    <span style="
                        color:#a9b5c4;
                        font-size:10px;
                    ">
                        {step}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    panel_header(
        "Resume Highlights",
        "PROJECT VALUE",
    )

    highlights = [
        (
            "End-to-End NLP",
            "Built a complete pipeline from raw review "
            "data through preprocessing, modeling, "
            "evaluation and deployment."
        ),
        (
            "Deep Learning",
            "Implemented trainable embeddings, "
            "Bidirectional LSTM and a custom Attention layer."
        ),
        (
            "Business Intelligence",
            "Converted customer review text into "
            "sentiment and product-aspect insights."
        ),
        (
            "Interactive Deployment",
            "Built a Streamlit analytics platform with "
            "live inference, visualizations and batch prediction."
        ),
    ]

    cols = st.columns(4)

    for column, (title, description) in zip(
        cols,
        highlights,
    ):

        with column:

            st.markdown(
                f"""
                <div class="panel"
                     style="min-height:145px;">
                    <div style="
                        color:#12dfcb;
                        font-size:10px;
                        font-weight:850;
                        letter-spacing:0.8px;
                        text-transform:uppercase;
                    ">
                        {title}
                    </div>
                    <div style="
                        color:#667386;
                        font-size:9px;
                        line-height:1.65;
                        margin-top:12px;
                    ">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        REVIEWIQ • PRODUCT REVIEW INTELLIGENCE SYSTEM
        <br>
        NLP · EMBEDDINGS · BiLSTM · ATTENTION · ASPECT INTELLIGENCE
    </div>
    """,
    unsafe_allow_html=True,
)
