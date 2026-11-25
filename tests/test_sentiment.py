"""Tests for the sentiment analysis module."""

from __future__ import annotations

from src.sentiment import finbert_sentiment, keyword_sentiment


def test_finbert_returns_valid_labels():
    results = finbert_sentiment(["Revenue increased significantly."])
    assert len(results) == 1
    assert results.iloc[0]["label"] in {"positive", "negative", "neutral"}


def test_finbert_probabilities_sum_to_one():
    results = finbert_sentiment(["The company reported a net loss."])
    row = results.iloc[0]
    total = row["positive"] + row["negative"] + row["neutral"]
    assert abs(total - 1.0) < 0.01


def test_finbert_batch_processing():
    texts = ["Good earnings.", "Bad results.", "No change."]
    results = finbert_sentiment(texts)
    assert len(results) == 3


def test_keyword_positive():
    results = keyword_sentiment(["Strong growth and profit increase."])
    assert results.iloc[0]["label"] == "positive"
    assert results.iloc[0]["n_positive_words"] > 0


def test_keyword_negative():
    results = keyword_sentiment(["Significant loss and decline in revenue."])
    assert results.iloc[0]["label"] == "negative"
    assert results.iloc[0]["n_negative_words"] > 0


def test_keyword_neutral():
    results = keyword_sentiment(["The meeting was held on Tuesday."])
    assert results.iloc[0]["label"] == "neutral"
    assert results.iloc[0]["score"] == 0.0
