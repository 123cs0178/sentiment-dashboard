"""
Fine-tune DistilBERT on Amazon Polarity reviews.
Run once: python train.py
Saves to ./model/
"""

import os, torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

MODEL_DIR = "./model"
BASE_MODEL = "distilbert-base-uncased"
LABELS = ["negative", "positive"]

# ── 1. Load dataset (amazon_polarity: 0=neg, 1=pos) ──────────────────────────
print("Loading dataset...")
dataset = load_dataset("amazon_polarity", split={"train": "train[:8000]", "test": "test[:2000]"})

# ── 2. Tokenize ───────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(batch["content"], truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True, remove_columns=["title", "content"])
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

# ── 3. Model ──────────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2, id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1}
)

# ── 4. Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# ── 5. Training args ──────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# ── 6. Train ──────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

print("Training...")
trainer.train()

# ── 7. Save ───────────────────────────────────────────────────────────────────
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")

# ── 8. Quick eval ─────────────────────────────────────────────────────────────
results = trainer.evaluate()
print(f"\nFinal → Accuracy: {results['eval_accuracy']:.4f} | F1: {results['eval_f1']:.4f}")
