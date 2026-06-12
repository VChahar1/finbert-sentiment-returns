from __future__ import annotations

import numpy as np
import pandas as pd


from .sentiment import finbert_sentiment, keyword_sentiment


def load_phrasebank() -> pd.DataFrame:
    """Load financial_phrasebank from the locally downloaded text file."""
    path = "data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"

    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.rfind("@")
            if idx == -1:
                continue
            text = line[:idx].strip()
            label = line[idx + 1:].strip().lower()
            if label in ("positive", "negative", "neutral"):
                rows.append({"text": text, "human_label": label})

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} sentences from Sentences_AllAgree.txt")
    return df

def compute_metrics(true_labels: pd.Series, pred_labels: pd.Series) -> dict:
    """Compute accuracy and per-class precision/recall."""
    classes = ["positive", "negative", "neutral"]
    correct = (true_labels == pred_labels).sum()
    total = len(true_labels)
    accuracy = correct / total

    metrics = {"accuracy": float(accuracy), "n_samples": total}
    for cls in classes:
        tp = ((pred_labels == cls) & (true_labels == cls)).sum()
        fp = ((pred_labels == cls) & (true_labels != cls)).sum()
        fn = ((pred_labels != cls) & (true_labels == cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f"{cls}_precision"] = float(precision)
        metrics[f"{cls}_recall"] = float(recall)
        metrics[f"{cls}_count"] = int((true_labels == cls).sum())

    return metrics


def confusion_matrix(true_labels: pd.Series, pred_labels: pd.Series) -> pd.DataFrame:
    """Build a confusion matrix as a DataFrame."""
    classes = ["positive", "negative", "neutral"]
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    for t, p in zip(true_labels, pred_labels):
        if t in classes and p in classes:
            matrix.loc[t, p] += 1
    matrix.index.name = "true \\ predicted"
    return matrix


def run_evaluation() -> dict:
    """Run the full evaluation pipeline."""
    print("Loading financial_phrasebank dataset...")
    data = load_phrasebank()
    texts = data["text"].tolist()
    true_labels = data["human_label"]

    print(f"Loaded {len(texts)} sentences")
    print(f"Label distribution: {true_labels.value_counts().to_dict()}")
    print()

    # FinBERT evaluation.
    print("Running FinBERT on all sentences...")
    fb_results = finbert_sentiment(texts)
    fb_metrics = compute_metrics(true_labels, fb_results["label"])
    fb_cm = confusion_matrix(true_labels, fb_results["label"])

    print(f"\nFinBERT accuracy: {fb_metrics['accuracy']:.1%}")
    print("Confusion matrix:")
    print(fb_cm)
    print()
    for cls in ["positive", "negative", "neutral"]:
        print(f"  {cls:>8s}: precision={fb_metrics[f'{cls}_precision']:.2%}, "
              f"recall={fb_metrics[f'{cls}_recall']:.2%} "
              f"(n={fb_metrics[f'{cls}_count']})")

    # Keyword baseline evaluation.
    print("\n" + "=" * 60)
    print("Running keyword baseline on all sentences...")
    kw_results = keyword_sentiment(texts)
    kw_metrics = compute_metrics(true_labels, kw_results["label"])
    kw_cm = confusion_matrix(true_labels, kw_results["label"])

    print(f"\nKeyword baseline accuracy: {kw_metrics['accuracy']:.1%}")
    print("Confusion matrix:")
    print(kw_cm)
    print()
    for cls in ["positive", "negative", "neutral"]:
        print(f"  {cls:>8s}: precision={kw_metrics[f'{cls}_precision']:.2%}, "
              f"recall={kw_metrics[f'{cls}_recall']:.2%} "
              f"(n={kw_metrics[f'{cls}_count']})")

    # Summary comparison.
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  FinBERT accuracy:  {fb_metrics['accuracy']:.1%}")
    print(f"  Keyword accuracy:  {kw_metrics['accuracy']:.1%}")
    print(f"  FinBERT advantage: {fb_metrics['accuracy'] - kw_metrics['accuracy']:+.1%}")

    return {
        "finbert": fb_metrics,
        "keyword": kw_metrics,
        "finbert_cm": fb_cm,
        "keyword_cm": kw_cm,
    }


if __name__ == "__main__":
    run_evaluation()