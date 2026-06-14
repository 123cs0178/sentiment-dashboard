from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from collections import deque
import time, torch, os

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "RahulBror/sentiment-dashboard-model"
device = 0 if torch.cuda.is_available() else -1
history: deque = deque(maxlen=100)

print(f"Loading model {MODEL_NAME}...")
try:
    clf = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=256,
    )
    MODEL_LOADED = True
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load failed: {e}")
    MODEL_LOADED = False

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

def make_record(text: str, result: dict) -> dict:
    label = result["label"].lower()
    score = result["score"]
    sentiment_score = score if "pos" in label else 1 - score
    return {
        "text": text[:300],
        "label": "positive" if "pos" in label else "negative",
        "confidence": round(score * 100, 2),
        "sentiment_score": round(sentiment_score * 100, 2),
        "timestamp": time.time(),
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model": MODEL_NAME,
        "model_loaded": MODEL_LOADED,
    }

@app.post("/analyze")
def analyze(body: TextInput):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded")
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    try:
        result = clf(body.text)[0]
        record = make_record(body.text, result)
        history.appendleft(record)
        return record
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

@app.post("/analyze/batch")
def analyze_batch(body: BatchInput):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded")
    if not body.texts:
        raise HTTPException(400, "texts list is empty")
    if len(body.texts) > 20:
        raise HTTPException(400, "Max 20 texts per batch")
    try:
        results = clf(body.texts)
        records = [make_record(t, r) for t, r in zip(body.texts, results)]
        for r in reversed(records):
            history.appendleft(r)
        return records
    except Exception as e:
        raise HTTPException(500, f"Batch inference error: {str(e)}")

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
