"""Generate figures for the README."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def figure_accuracy_comparison(fb_acc: float, kw_acc: float) -> None:
    """Bar chart comparing FinBERT vs keyword baseline accuracy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["FinBERT", "Keyword\nbaseline"]
    accs = [fb_acc * 100, kw_acc * 100]
    colors = ["steelblue", "lightcoral"]
    bars = ax.bar(methods, accs, color=colors, width=0.5, edgecolor="black")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Sentiment classification accuracy on financial_phrasebank")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/accuracy_comparison.png", dpi=120)
    print("Saved figures/accuracy_comparison.png")


def figure_confusion_matrix(cm: pd.DataFrame, title: str, filename: str) -> None:
    """Heatmap of a confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=cm.columns, yticklabels=cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"figures/{filename}", dpi=120)
    print(f"Saved figures/{filename}")


if __name__ == "__main__":
    # Run the evaluation first to get numbers.
    from src.evaluate import run_evaluation
    results = run_evaluation()

    figure_accuracy_comparison(
        results["finbert"]["accuracy"],
        results["keyword"]["accuracy"],
    )
    figure_confusion_matrix(
        results["finbert_cm"],
        "FinBERT confusion matrix",
        "finbert_cm.png",
    )
    figure_confusion_matrix(
        results["keyword_cm"],
        "Keyword baseline confusion matrix",
        "keyword_cm.png",
    )