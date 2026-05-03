
import streamlit as st
import joblib
import numpy as np
import re

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# LOAD ML MODEL
# -----------------------------
@st.cache_resource
def load_ml_model():
    model = joblib.load("ml_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

ml_model, vectorizer = load_ml_model()

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# ML PREDICTION
# -----------------------------
def predict_text_ml(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred = ml_model.predict(vec)[0]
    probs = ml_model.predict_proba(vec)[0]
    confidence = np.max(probs)

    ai_prob = probs[1]
    human_prob = probs[0]

    return pred, confidence, ai_prob, human_prob

# -----------------------------
# LABEL MAP
# -----------------------------
def label_map(pred):
    if pred == 1:
        return "Likely AI Generated 🤖"
    else:
        return "Likely Human Written 👤"

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Info")
st.sidebar.write("**Project:** AI vs Human Text Detector")
st.sidebar.write("**Task:** Binary Text Classification")
st.sidebar.write("**Classes:** AI / Human")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Details")
st.sidebar.write("**Model:** TF-IDF + Naive Bayes")
st.sidebar.write("**Type:** Machine Learning")

# -----------------------------
# MAIN PAGE
# -----------------------------
st.title("🧠 AI vs Human Text Detector")
st.markdown(
    "Enter any text and detect whether it is **AI-generated** or **human-written** using a **Machine Learning model**."
)

user_input = st.text_area(
    "📝 Enter Text Here:",
    height=220,
    placeholder="Paste any paragraph, essay, sentence, or social media text..."
)

if st.button("🔍 Detect Text Source", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        pred, conf, ai_prob, human_prob = predict_text_ml(user_input)

        result = label_map(pred)

        st.subheader("🎯 Prediction Result")

        if pred == 1:
            st.error(result)
        else:
            st.success(result)

        st.info(f"Confidence: {conf*100:.2f}%")
        st.write("**Model Used:** TF-IDF + Naive Bayes (ML)")
        st.write(f"**AI Probability:** {ai_prob*100:.2f}%")
        st.write(f"**Human Probability:** {human_prob*100:.2f}%")

        st.progress(int(conf * 100))

# -----------------------------
# EXAMPLES
# -----------------------------
st.markdown("---")
st.subheader("💡 Example Texts")

examples = [
    "In conclusion, artificial intelligence has revolutionized multiple industries and continues to shape the future.",
    "lol this is crazy",
    "The weather was nice so I just went outside and sat there for a bit.",
    "The implementation of neural architectures has significantly advanced natural language understanding."
]

for ex in examples:
    st.code(ex)

# -----------------------------
# MODEL INFO
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Information")

info_table = {
    "Feature": ["Algorithm Type", "Speed", "Context Understanding", "Complexity"],
    "ML Model": ["Traditional ML", "Fast", "Moderate", "Low"]
}

st.table(info_table)

# -----------------------------
# ABOUT
# -----------------------------
st.markdown("---")
st.subheader("📚 About This Project")
st.write("""
This project detects whether a text is:
- **AI Generated**
- **Human Written**

Model used:
- **TF-IDF + Naive Bayes**

Useful in:
- academic integrity checks
- content moderation
- writing analysis
- AI text detection systems
""")
