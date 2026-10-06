# Four-Class-Tweet-Sentiment-Classification-Using-Deep-Learning

## Live App

Try the deployed app here:

[Open the Tweet Sentiment Analysis app](https://tweet-sentiment-analysis-using-nlpgit-wmatjmdzndds6undgnvfg9.streamlit.app/)

## Project Overview

People share opinions about products, services, and events through tweets. Since tweets are short, informal, and posted in large numbers, analyzing them manually can be difficult.

This project uses natural language processing and deep learning to classify tweets into four categories: **Positive, Negative, Neutral, and Irrelevant**. It explores LSTM, BiLSTM, and GRU models with FastText word embeddings.

## Objectives

- Preprocess tweet text for classification.
- Represent words using FastText embeddings.
- Train and compare LSTM, BiLSTM, and GRU models.
- Evaluate the models using accuracy, loss, classification reports, and confusion matrices.
- Provide a Streamlit app for classifying new text.

## Sentiment Categories

| Category | Meaning |
| Positive | The tweet expresses a favorable opinion or emotion. |
| Negative | The tweet expresses an unfavorable opinion or emotion. |
| Neutral  | The tweet is relevant but has no clear positive or negative opinion. |
| Irrelevant | The tweet does not meaningfully relate to the sentiment task. |

## Models

- **LSTM** — learns patterns across sequences of words.
- **BiLSTM** — processes text in both forward and backward directions.
- **GRU** — uses a gated recurrent architecture to learn sequence patterns.

FastText embeddings provide word representations for the models. Results depend on the training data, preprocessing, model settings, and evaluation split.

## Repository Files

Your repository may include:

```text
.
├── app.py
├── requirements.txt
├── model.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── config.json
├── training_notebook.ipynb
└── README.md
