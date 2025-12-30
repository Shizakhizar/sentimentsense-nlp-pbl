# SentimentSense — NLP PBL Project (Interactive Sentiment Analysis)

## NLP Category
Text Classification & Tagging — Sentiment Analysis (Binary)

## Task
Given a user’s input text, predict sentiment (Positive/Negative) with confidence scores.

## Tech Stack
- Python
- HuggingFace Transformers + Datasets
- PyTorch
- Streamlit (Interactive UI)
- Scikit-learn (Metrics)

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
### 2) Train the model
python scripts/train.py

### 3) Run the interactive app
streamlit run app/app.py