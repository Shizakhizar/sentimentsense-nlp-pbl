import streamlit as st
from transformers import pipeline
import spacy

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SentimentSense+",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 SentimentSense+")
st.subheader("Multi-Task NLP System (Text Classification & Tagging)")

st.markdown("""
This interactive system demonstrates **multiple NLP tasks** under **one NLP category**:
**Text Classification & Tagging**.

**Core Task:** Sentiment Analysis  
**Additional Tasks:** Toxic Content Detection, Topic Classification, Part-of-Speech Tagging
""")

# ---------------- MODEL LOADERS (ON DEMAND) ----------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_toxic_model():
    return pipeline("text-classification", model="unitary/toxic-bert")

@st.cache_resource
def load_topic_model():
    return pipeline("zero-shot-classification")

@st.cache_resource
def load_pos_model():
    return spacy.load("en_core_web_sm")

# ---------------- USER INPUT ----------------
text = st.text_area("Enter text:", height=120)

task = st.selectbox(
    "Select NLP Task",
    [
        "Sentiment Analysis",
        "Toxic Content Detection",
        "Topic Classification",
        "Part-of-Speech Tagging"
    ]
)

# ---------------- RUN BUTTON ----------------
if st.button("Run NLP Task"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # -------- SENTIMENT ANALYSIS --------
        if task == "Sentiment Analysis":
            with st.spinner("Loading sentiment model..."):
                model = load_sentiment_model()
            result = model(text)[0]
            st.success(f"Sentiment: {result['label']}")
            st.info(f"Confidence: {result['score']:.2f}")

        # -------- TOXIC CONTENT --------
        elif task == "Toxic Content Detection":
            with st.spinner("Loading toxicity model..."):
                model = load_toxic_model()
            result = model(text)[0]
            st.success(f"Toxicity Label: {result['label']}")
            st.info(f"Confidence: {result['score']:.2f}")

        # -------- TOPIC CLASSIFICATION --------
        elif task == "Topic Classification":
            with st.spinner("Loading topic classifier..."):
                model = load_topic_model()
            candidate_labels = [
                "technology",
                "education",
                "politics",
                "sports",
                "business",
                "health",
                "entertainment"
            ]
            result = model(text, candidate_labels)
            st.success(f"Predicted Topic: {result['labels'][0]}")
            st.info(f"Confidence: {result['scores'][0]:.2f}")

        # -------- POS TAGGING --------
        elif task == "Part-of-Speech Tagging":
            with st.spinner("Loading POS tagger..."):
                nlp = load_pos_model()
            doc = nlp(text)
            st.subheader("POS Tags")
            for token in doc:
                st.write(f"**{token.text}** → {token.pos_}")
