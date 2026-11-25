from __future__ import annotations

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


_finbert_model = None
_finbert_tokenizer = None


def _load_finbert():
    """Load FinBERT model and tokenizer (cached after first call)."""
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None:
        print("Loading FinBERT model...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )
        _finbert_model.eval()
    return _finbert_model, _finbert_tokenizer


def finbert_sentiment(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    """Return FinBERT labels and probabilities, loading public weights on demand."""
    model, tokenizer = _load_finbert()
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()

        for j, text in enumerate(batch):
            label_idx = int(probs[j].argmax())
            all_results.append(
                {
                    "text": text,
                    "label": label_map[label_idx],
                    "positive": float(probs[j][0]),
                    "negative": float(probs[j][1]),
                    "neutral": float(probs[j][2]),
                }
            )

    return pd.DataFrame(all_results)


LM_POSITIVE = {
    "achieve",
    "gain",
    "improve",
    "profit",
    "success",
    "strong",
    "growth",
    "positive",
    "increase",
    "better",
    "benefit",
    "opportunity",
    "upgrade",
    "outperform",
    "surpass",
    "exceed",
    "favorable",
    "optimistic",
    "recovery",
    "innovation",
    "efficient",
    "advancement",
    "superior",
    "attract",
}

LM_NEGATIVE = {
    "loss",
    "decline",
    "risk",
    "fail",
    "weak",
    "negative",
    "decrease",
    "deficit",
    "litigation",
    "impairment",
    "restructuring",
    "downturn",
    "adverse",
    "deteriorate",
    "penalty",
    "violation",
    "default",
    "lawsuit",
    "bankruptcy",
    "fraud",
    "concern",
    "underperform",
    "disappoint",
    "worst",
}


def keyword_sentiment(texts: list[str]) -> pd.DataFrame:
    """Classify texts using the included small positive and negative word sets."""
    results = []
    for text in texts:
        words = set(text.lower().split())
        n_pos = len(words & LM_POSITIVE)
        n_neg = len(words & LM_NEGATIVE)
        n_total = max(len(words), 1)
        score = (n_pos - n_neg) / n_total

        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        results.append(
            {
                "text": text,
                "label": label,
                "score": score,
                "n_positive_words": n_pos,
                "n_negative_words": n_neg,
            }
        )
    return pd.DataFrame(results)


def _demo() -> None:
    samples = [
        "Revenue increased 15% year over year driven by strong demand.",
        "The company reported a net loss of $2.3 billion for the quarter.",
        "The board approved a regular quarterly dividend of $0.50 per share.",
        "Restructuring charges and impairment losses weighed on earnings.",
        "Market conditions remained stable with no significant changes.",
    ]

    print("FinBERT sentiment scoring")
    print("=" * 70)
    fb = finbert_sentiment(samples)
    for _, row in fb.iterrows():
        print(
            f"  [{row['label']:>8s}] (pos={row['positive']:.2f} "
            f"neg={row['negative']:.2f} neu={row['neutral']:.2f})  "
            f"{row['text'][:60]}..."
        )

    print()
    print("Keyword baseline")
    print("=" * 70)
    kw = keyword_sentiment(samples)
    for _, row in kw.iterrows():
        print(
            f"  [{row['label']:>8s}] (score={row['score']:+.3f})  {row['text'][:60]}..."
        )


if __name__ == "__main__":
    _demo()
