# ==========================================================
# Fake News Detector - Streamlit Web App
#
# This uses your ALREADY TRAINED model (fake_news_model.pkl
# and tfidf.pkl). It does NOT retrain anything.
#
# HOW TO RUN:
#   1. pip install streamlit
#   2. streamlit run app.py
#   3. Your browser will open automatically at http://localhost:8501
#
# Anyone on the same WiFi/network can also access it using the
# "Network URL" that Streamlit prints in the terminal.
# ==========================================================

import re
import string
import joblib
import streamlit as st
import nltk
import plotly.graph_objects as go

# --- Fix for Windows SSL certificate errors ---
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ------------------------------------------------

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

MODEL_TEST_ACCURACY = 95.25  # Linear SVM -- update this if you retrain with different results


def clean_text(text):
    """Same cleaning pipeline used during training -- must match exactly."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf.pkl")
    return model, vectorizer


def predict_news(text, model, vectorizer):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    confidence = None
    fake_pct = None
    real_pct = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        fake_pct = round(proba[0] * 100, 2)
        real_pct = round(proba[1] * 100, 2)
        confidence = max(fake_pct, real_pct)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vector)[0]
        confidence = round(min(abs(score) * 20 + 50, 99.9), 2)
        if result == "REAL NEWS":
            real_pct = confidence
            fake_pct = round(100 - confidence, 2)
        else:
            fake_pct = confidence
            real_pct = round(100 - confidence, 2)

    return result, confidence, fake_pct, real_pct


# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="TruthLens",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 TruthLens")
st.caption("See Through the Noise. Find the Truth.")
st.write(
    "Paste a news headline or article below to check whether it's likely "
    "**REAL** or **FAKE**, using a machine learning model trained on "
    "72,000+ real-world news articles (WELFake dataset)."
)

st.info(f"📊 Model: **Linear SVM**  |  Overall Test Accuracy: **{MODEL_TEST_ACCURACY}%**")

# Load model (cached so it only loads once, not on every interaction)
try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Could not find the trained model files. Make sure "
        "`fake_news_model.pkl` and `tfidf.pkl` exist "
        "in the same folder as this app (repository root), and that "
        "you've already run your training script (fake_news_real.py) "
        "at least once."
    )
    st.stop()

# ==========================================================
# Input Area
# ==========================================================

news_text = st.text_area(
    "Enter news text here:",
    height=180,
    placeholder="e.g. Scientists announce breakthrough treatment after successful clinical trials..."
)

col1, col2 = st.columns([1, 3])
with col1:
    check_button = st.button("🔍 Check News", type="primary", use_container_width=True)

if check_button:
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        with st.spinner("Analyzing..."):
            result, confidence, fake_pct, real_pct = predict_news(news_text, model, vectorizer)

        st.markdown("---")

        # ---- Big colored result banner ----
        if result == "REAL NEWS":
            st.success(f"✅ Prediction: **{result}**")
            gauge_color = "#2ecc71"
        else:
            st.error(f"⚠️ Prediction: **{result}**")
            gauge_color = "#e74c3c"

        # ---- Gauge chart for confidence ----
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Prediction Confidence", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 50], "color": "#fdecea"},
                    {"range": [50, 75], "color": "#fff4e5"},
                    {"range": [75, 100], "color": "#e8f8f0"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": confidence,
                },
            },
        ))
        gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # ---- Bar chart: FAKE vs REAL probability ----
        bar_fig = go.Figure(data=[
            go.Bar(
                x=["FAKE NEWS", "REAL NEWS"],
                y=[fake_pct, real_pct],
                marker_color=["#e74c3c", "#2ecc71"],
                text=[f"{fake_pct}%", f"{real_pct}%"],
                textposition="outside",
            )
        ])
        bar_fig.update_layout(
            title="Class Probability Breakdown",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 110],
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # ---- Model accuracy metric ----
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("This Prediction's Confidence", f"{confidence}%")
        with col_b:
            st.metric("Model's Overall Test Accuracy", f"{MODEL_TEST_ACCURACY}%")

        st.caption(
            "Confidence reflects how strongly the model leans toward its "
            "prediction for THIS specific input. Overall accuracy reflects "
            "performance across 14,427 held-out test articles during training."
        )

st.markdown("---")
st.caption(
    "🔍 TruthLens | Built with scikit-learn + NLTK + Streamlit | "
    "Trained on the WELFake real-world news dataset (72,134 articles)"
)
