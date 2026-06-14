from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from collections import deque
import time, os

HISTORY_MAX = 100

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN    
)

print(f"Using Hugging Face Inference API for {MODEL_ID}")
history: deque = deque(maxlen=HISTORY_MAX)

# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_record(text: str, result: dict) -> dict:
    label = result["label"]          # "positive" | "negative"
    score = result["score"]
    # Normalize to [0,1] where 1 = most positive
    sentiment_score = score if label == "positive" else 1 - score
    return {
        "text": text[:300],
        "label": label,
        "confidence": round(score * 100, 2),
        "sentiment_score": round(sentiment_score * 100, 2),
        "timestamp": time.time(),
    }


def predict_sentiment(text):
    result = client.text_classification(
    text,
    model=MODEL_ID,
    provider="hf-inference"
    )
    return {
        "label": result.label.lower(),
        "score": result.score
    }
# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID
    }

@app.post("/analyze")
def analyze(body: TextInput):
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    result = predict_sentiment(body.text)
    record = make_record(body.text, result)
    history.appendleft(record)
    return record

@app.post("/analyze/batch")
def analyze_batch(body: BatchInput):
    if not body.texts:
        raise HTTPException(400, "texts list is empty.")
    if len(body.texts) > 20:
        raise HTTPException(400, "Max 20 texts per batch.")
    results = [predict_sentiment(text) for text in body.texts]   
    records = [make_record(t, r) for t, r in zip(body.texts, results)]
    for r in reversed(records):
        history.appendleft(r)
    return records

@app.get("/history")
def get_history():
    return list(history)

@app.get("/analytics")
def analytics():
    if not history:
        return {"total": 0}
    items = list(history)
    total = len(items)
    pos = sum(1 for i in items if i["label"] == "positive")
    neg = total - pos
    avg_score = sum(i["sentiment_score"] for i in items) / total
    avg_conf = sum(i["confidence"] for i in items) / total

    # Score distribution buckets
    buckets = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for i in items:
        s = i["sentiment_score"]
        if s < 20: buckets["0-20"] += 1
        elif s < 40: buckets["20-40"] += 1
        elif s < 60: buckets["40-60"] += 1
        elif s < 80: buckets["60-80"] += 1
        else: buckets["80-100"] += 1

    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_pct": round(pos / total * 100, 1),
        "avg_sentiment_score": round(avg_score, 1),
        "avg_confidence": round(avg_conf, 1),
        "distribution": buckets,
    }

@app.delete("/history")
def clear_history():
    history.clear()
    return {"status": "cleared"}


