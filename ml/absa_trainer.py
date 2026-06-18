"""
Aspect-Based Sentiment Analysis — BERT fine-tuning.
Dataset format: {text, aspect, sentiment} where sentiment ∈ {positive, negative, neutral}
Run: python scripts/train_models.py --model absa
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
BASE_MODEL = "bert-base-uncased"


class ABSADataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }


def prepare_data(reviews_csv: str) -> tuple[list, list]:
    """
    Prepare ABSA training data from reviews CSV.
    Expected columns: review_text, aspect, sentiment
    If not available, uses weak supervision to auto-label.
    """
    df = pd.read_csv(reviews_csv)

    if "aspect" not in df.columns:
        # Weak supervision: create aspect-sentiment pairs using keywords
        logger.info("No aspect column — using weak supervision")
        aspect_keywords = {
            "battery": ["battery", "charge", "charging", "power", "drain"],
            "camera": ["camera", "photo", "picture", "image", "video", "lens"],
            "sound": ["sound", "audio", "bass", "treble", "volume", "speaker"],
            "price": ["price", "cost", "value", "worth", "expensive", "cheap"],
            "build": ["build", "quality", "material", "plastic", "metal", "sturdy"],
        }
        rows = []
        for _, row in df.iterrows():
            text = str(row.get("reviewText", "")).lower()
            rating = float(row.get("overall", 3.0))
            sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
            for aspect, keywords in aspect_keywords.items():
                if any(kw in text for kw in keywords):
                    rows.append({
                        "text": f"[{aspect}] {row.get('reviewText', '')}",
                        "sentiment": sentiment
                    })
        df = pd.DataFrame(rows)
    else:
        df["text"] = df.apply(
            lambda r: f"[{r['aspect']}] {r['review_text']}", axis=1
        )

    texts = df["text"].tolist()
    labels = [LABEL2ID.get(s, 1) for s in df["sentiment"].tolist()]
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}


def train(reviews_csv: str = None, output_dir: str = None):
    reviews_csv = reviews_csv or settings.reviews_csv
    output_dir = output_dir or settings.absa_model_path

    logger.info(f"Training ABSA model from {reviews_csv}")
    texts, labels = prepare_data(reviews_csv)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_dataset = ABSADataset(X_train, y_train, tokenizer)
    val_dataset = ABSADataset(X_val, y_val, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        report_to="mlflow",
        run_name="absa_bert"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("Starting ABSA training...")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"ABSA model saved → {output_dir}")

    # Final evaluation
    results = trainer.evaluate(val_dataset)
    logger.info(f"Val F1 macro: {results['eval_f1_macro']:.4f}")
    return results


if __name__ == "__main__":
    train()
