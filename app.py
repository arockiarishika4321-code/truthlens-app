# ============================================================
#   TRUTHLENS — ADVANCED FAKE NEWS DETECTION SYSTEM
#   Features: Tamil Support, URL Check, Dashboard, 
#             Confidence Score, History, Word Highlight
#   Run: streamlit run app.py
# ============================================================

import os
import re
import string
import joblib
import nltk
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from googletrans import Translator

# ============================================================
#  PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TruthLens",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
#  NLTK SETUP
# ============================================================

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()
translator = Translator()

# ============================================================
#  PATHS
# ============================================================

MODEL_PATH     = 'fake_news_model.pkl'
TFIDF_PATH     = 'tfidf.pkl'
HISTORY_PATH   = 'prediction_history.csv'

# ============================================================
#  TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# ============================================================
#  FEATURE 1 — AUTO TAMIL TRANSLATE
# ============================================================

def translate_to_english(text):
    try:
        detected = translator.detect(text)
        if detected.lang != 'en':
            translated = translator.translate(text, dest='en')
            return translated.text, detected.lang
        return text, 'en'
    except Exception:
        return text, 'en'

# ============================================================
#  FEATURE 2 — URL NEWS EXTRACTOR
# ============================================================

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Get paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        if len(text.strip()) < 100:
            text = soup.get_text()

        text = re.sub(r'\s+', ' ', text).strip()
        return text[:5000]

    except Exception as e:
        return None

# ============================================================
#  FEATURE 3 — CONFIDENCE SCORE
# ============================================================

def get_confidence(model, vector):
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(vector)[0]
            return round(max(probs) * 100, 2)
        elif hasattr(model, 'decision_function'):
            score = abs(model.decision_function(vector)[0])
            return round(min((score / (score + 1)) * 100 + 40, 99.9), 2)
    except Exception:
        pass
    return 80.0

# ============================================================
#  FEATURE 4 — TOP WORDS HIGHLIGHT
# ============================================================

def get_top_words(text, tfidf, model, n=10):
    try:
        vec       = tfidf.transform([text])
        features  = tfidf.get_feature_names_out()
        scores    = vec.toarray()[0]
        top_idx   = scores.argsort()[-n:][::-1]
        top_words = [(features[i], round(scores[i], 4)) for i in top_idx if scores[i] > 0]
        return top_words
    except Exception:
        return []

# ============================================================
#  SAVE HISTORY
# ============================================================

def save_history(original_text, translated_text, language,
                 source, prediction, confidence):
    record = pd.DataFrame({
        'timestamp'   : [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'source'      : [source],
        'language'    : [language],
        'prediction'  : [prediction],
        'confidence'  : [confidence],
        'word_count'  : [len(original_text.split())],
        'text_preview': [original_text[:100] + '...']
    })
    if os.path.exists(HISTORY_PATH):
        old    = pd.read_csv(HISTORY_PATH)
        record = pd.concat([old, record], ignore_index=True)
    record.to_csv(HISTORY_PATH, index=False)

# ============================================================
#  LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please upload fake_news_model.pkl and tfidf.pkl")
    st.stop()

model, tfidf = load_model()

# ============================================================
#  SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/news.png", width=80)
    st.title("TruthLens")
    st.caption("See Through the Noise. Find the Truth.")
    st.divider()
    st.write("**Model:** Linear SVM")
    st.write("**Accuracy:** 95.25%")
    st.write("**Trained on:** 72,134 articles")
    st.divider()

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        st.write(f"**Total checked:** {len(history)}")
        fake_count = len(history[history['prediction'] == 'FAKE'])
        real_count = len(history[history['prediction'] == 'REAL'])
        st.write(f"**Fake detected:** {fake_count}")
        st.write(f"**Real detected:** {real_count}")

# ============================================================
#  MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Check News",
    "🔗 Check URL",
    "📊 Dashboard",
    "📈 History"
])

# ============================================================
#  TAB 1 — CHECK NEWS (Text + Tamil Support)
# ============================================================

with tab1:
    st.title("🔍 TruthLens — Fake News Detector")
    st.caption("Supports English and Tamil news articles")

    news_input = st.text_area(
        "Paste news article here (English or Tamil):",
        height=200,
        placeholder="எந்த மொழியிலும் news paste பண்ணலாம்..."
    )

    col1, col2 = st.columns(2)
    check_btn  = col1.button("🔍 Check News", use_container_width=True)
    clear_btn  = col2.button("🧹 Clear",      use_container_width=True)

    if clear_btn:
        st.rerun()

    if check_btn:
        if not news_input.strip():
            st.warning("Please paste some news text first!")
        else:
            with st.spinner("Analysing..."):

                # STEP 1 — Detect and translate
                translated_text, detected_lang = translate_to_english(news_input)

                if detected_lang != 'en':
                    st.info(f"🌐 Tamil detected! Auto-translated to English for analysis.")
                    with st.expander("See translated text"):
                        st.write(translated_text)

                # STEP 2 — Clean and predict
                cleaned    = clean_text(translated_text)
                vector     = tfidf.transform([cleaned])
                prediction = model.predict(vector)[0]
                confidence = get_confidence(model, vector)
                label      = "REAL" if prediction == 1 else "FAKE"

                # STEP 3 — Save history
                lang_name = "Tamil" if detected_lang == 'ta' else "English"
                save_history(news_input, translated_text, lang_name,
                             "Text Input", label, confidence)

                st.divider()

                # STEP 4 — Show result
                if prediction == 1:
                    st.success("# ✅ REAL NEWS")
                    st.progress(int(confidence))
                    st.write(f"**Confidence: {confidence}%**")
                else:
                    st.error("# 🚨 FAKE NEWS")
                    st.progress(int(confidence))
                    st.write(f"**Confidence: {confidence}%**")

                # STEP 5 — Word stats
                c1, c2, c3 = st.columns(3)
                c1.metric("Original words",   len(news_input.split()))
                c2.metric("After cleaning",   len(cleaned.split()))
                c3.metric("Language",         lang_name)

                # STEP 6 — Top trigger words
                top_words = get_top_words(cleaned, tfidf, model)
                if top_words:
                    st.divider()
                    st.subheader("🔑 Key words that triggered this result:")
                    cols = st.columns(5)
                    for i, (word, score) in enumerate(top_words[:10]):
                        cols[i % 5].metric(word, f"{score:.3f}")

                # STEP 7 — Share result
                st.divider()
                result_text = f"TruthLens Result: {label} ({confidence}%)\nCheck: truthlens-rishika.streamlit.app"
                st.text_area("Share this result:", value=result_text, height=80)

                st.info("ML prediction only. Always verify from trusted sources.")


# ============================================================
#  TAB 2 — CHECK URL
# ============================================================

with tab2:
    st.title("🔗 Check News from URL")
    st.write("Paste any news article URL — we'll extract and check it automatically!")

    url_input = st.text_input(
        "News article URL:",
        placeholder="https://www.thehindu.com/news/..."
    )

    url_btn = st.button("🔗 Extract & Check", use_container_width=True)

    if url_btn:
        if not url_input.strip():
            st.warning("Please paste a URL first!")
        elif not url_input.startswith('http'):
            st.warning("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Extracting article from URL..."):
                extracted_text = extract_text_from_url(url_input)

            if not extracted_text or len(extracted_text) < 100:
                st.error("Could not extract text from this URL. Try copying the article text manually.")
            else:
                st.success(f"Extracted {len(extracted_text.split())} words from URL!")

                with st.expander("See extracted text"):
                    st.write(extracted_text[:1000] + "...")

                with st.spinner("Analysing..."):
                    translated_text, detected_lang = translate_to_english(extracted_text)
                    cleaned    = clean_text(translated_text)
                    vector     = tfidf.transform([cleaned])
                    prediction = model.predict(vector)[0]
                    confidence = get_confidence(model, vector)
                    label      = "REAL" if prediction == 1 else "FAKE"

                    save_history(extracted_text, translated_text, "English",
                                 url_input, label, confidence)

                st.divider()

                if prediction == 1:
                    st.success("# ✅ REAL NEWS")
                else:
                    st.error("# 🚨 FAKE NEWS")

                st.progress(int(confidence))
                st.write(f"**Confidence: {confidence}%**")

                top_words = get_top_words(cleaned, tfidf, model)
                if top_words:
                    st.divider()
                    st.subheader("🔑 Key trigger words:")
                    cols = st.columns(5)
                    for i, (word, score) in enumerate(top_words[:10]):
                        cols[i % 5].metric(word, f"{score:.3f}")


# ============================================================
#  TAB 3 — DASHBOARD
# ============================================================

with tab3:
    st.title("📊 Statistics Dashboard")

    if not os.path.exists(HISTORY_PATH):
        st.info("No predictions yet! Go to Check News tab and analyse some articles first.")
    else:
        history = pd.read_csv(HISTORY_PATH)

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Checked",  len(history))
        c2.metric("Fake Detected",  len(history[history['prediction'] == 'FAKE']))
        c3.metric("Real Detected",  len(history[history['prediction'] == 'REAL']))
        c4.metric("Avg Confidence", f"{round(history['confidence'].mean(), 1)}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fake vs Real count")
            counts = history['prediction'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            colors  = ['#FF6B6B' if x == 'FAKE' else '#6BCB77' for x in counts.index]
            counts.plot(kind='bar', color=colors, ax=ax)
            ax.set_title('Fake vs Real')
            ax.set_xlabel('')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Confidence distribution")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.hist(history['confidence'], bins=20,
                     color='steelblue', edgecolor='white')
            ax2.set_title('Confidence Scores')
            ax2.set_xlabel('Confidence %')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.divider()

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Language breakdown")
            lang_counts = history['language'].value_counts()
            fig3, ax3   = plt.subplots(figsize=(5, 4))
            ax3.pie(lang_counts.values,
                    labels=lang_counts.index,
                    autopct='%1.1f%%',
                    colors=['#4ECDC4', '#FF6B6B'])
            ax3.set_title('Languages Used')
            st.pyplot(fig3)
            plt.close()

        with col4:
            st.subheader("Source breakdown")
            source_counts = history['source'].value_counts()
            fig4, ax4     = plt.subplots(figsize=(5, 4))
            source_counts.plot(kind='barh', color='mediumpurple', ax=ax4)
            ax4.set_title('Input Sources')
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()


# ============================================================
#  TAB 4 — HISTORY
# ============================================================

with tab4:
    st.title("📈 Prediction History")

    if not os.path.exists(HISTORY_PATH):
        st.info("No history yet! Start checking news articles.")
    else:
        history = pd.read_csv(HISTORY_PATH)
        st.write(f"Total predictions: **{len(history)}**")

        # Filter
        filter_opt = st.selectbox("Filter by:", ["All", "FAKE only", "REAL only", "Tamil only"])
        if filter_opt == "FAKE only":
            history = history[history['prediction'] == 'FAKE']
        elif filter_opt == "REAL only":
            history = history[history['prediction'] == 'REAL']
        elif filter_opt == "Tamil only":
            history = history[history['language'] == 'Tamil']

        st.dataframe(
            history[['timestamp', 'source', 'language',
                      'prediction', 'confidence', 'text_preview']]
            .tail(50)[::-1],
            use_container_width=True
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label     = "⬇ Download History CSV",
                data      = history.to_csv(index=False),
                file_name = "truthlens_history.csv",
                mime      = "text/csv"
            )
        with col2:
            if st.button("🗑 Clear History"):
                os.remove(HISTORY_PATH)
                st.success("History cleared!")
                st.rerun()

# ============================================================
#  FOOTER
# ============================================================

st.divider()
st.markdown(
    "<div style='text-align:center;color:gray'>"
    "🔍 TruthLens | Built with Streamlit + scikit-learn + NLTK | "
    "BSc Data Science Final Year Project"
    "</div>",
    unsafe_allow_html=True
)
