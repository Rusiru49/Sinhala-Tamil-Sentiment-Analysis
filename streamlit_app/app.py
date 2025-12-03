import streamlit as st
from inference.predictor import SentimentPredictor
from utils.logger import get_logger
import yaml

logger = get_logger("streamlit_app")

st.set_page_config(page_title="Sinhala/English Sentiment", layout="wide")

st.title("Sinhala & English Sentiment Classifier")
st.markdown("Model: **xlm-roberta-base** fine-tuned on HF datasets.")

@st.cache_resource
def load_predictor():
    return SentimentPredictor(model_dir="checkpoints/best_model")

predictor = load_predictor()

text = st.text_area("Enter text (Sinhala or English)", height=150)
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        preds, probs = predictor.predict(text)
        pred = preds[0]
        prob = probs[0]
        st.subheader("Prediction")
        st.write(f"Predicted label id: **{pred}**")
        st.write("Class probabilities:")
        st.dataframe({f"class_{i}": float(p) for i, p in enumerate(prob)}, width=400)
