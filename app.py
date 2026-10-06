import pickle
import re
from pathlib import Path

import contractions
import emoji
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bilstm_model.keras"
TOKENIZER_PATH = APP_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = APP_DIR / "label_encoder.pkl"
MAX_LEN = 50  # Same maxlen, padding, and truncation as the validation code


def clean_text(text):
    """Apply the same text cleanup steps as the user's validation code."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = contractions.fix(text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = emoji.demojize(text).replace(":", "")
    text = re.sub(r"([!?.])\1+", r"\1", text)
    text = text.lower()
    text = re.sub(r"https?:\S*|pic\.twitter\.com\S*", "", text)
    text = re.sub(r"https?:\S+|pic\.twitter\.com\S*", "", text)
    text = re.sub(r"(?<!\w)@(?=\s|$)", "", text)
    text = re.sub(r"[^\w\s!?:]", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(?:youtube|twitter|facebook|swaggiedeals|factory)\s+com\b", "", text)
    text = re.sub(r"\b[\w.-]+\s+com\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_artifacts():
    missing = [
        path.name
        for path in (MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    model = load_model(MODEL_PATH, compile=False)
    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)
    with LABEL_ENCODER_PATH.open("rb") as file:
        label_encoder = pickle.load(file)
    return model, tokenizer, label_encoder


st.set_page_config(page_title="BiLSTM Sentiment Classifier", page_icon="💬")
st.title("💬 BiLSTM Sentiment Classifier")
st.write("Enter a tweet or other text to predict its sentiment.")

try:
    model, tokenizer, label_encoder = load_artifacts()
except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.info("Place model.keras, tokenizer.pkl, and label_encoder.pkl beside app.py.")
    st.stop()

text_input = st.text_area("Text to classify", height=140)

if st.button("Predict sentiment", type="primary", disabled=not text_input.strip()):
    processed = clean_text(text_input)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    scores = np.asarray(model.predict(padded, verbose=0))[0].reshape(-1)
    if len(scores) != len(label_encoder.classes_):
        st.error(
            f"Model returned {len(scores)} class scores, but label_encoder has "
            f"{len(label_encoder.classes_)} labels. Check the model and encoder."
        )
        st.stop()

    predicted_index = int(np.argmax(scores))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(scores[predicted_index]) * 100

    st.subheader(f"Predicted sentiment: {predicted_label}")
    st.metric("Confidence", f"{confidence:.2f}%")
    with st.expander("Show preprocessing and class scores"):
        st.write("**Processed text**")
        st.code(processed or "(empty after preprocessing)")
        st.write("**Scores by class**")
        st.bar_chart({str(label): float(score) for label, score in zip(label_encoder.classes_, scores)})
