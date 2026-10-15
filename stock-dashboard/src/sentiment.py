"""
sentiment.py
============
Runs FinBERT (ProsusAI/finbert) on news headlines to produce:
  - Per-headline: label ('positive'/'neutral'/'negative') + confidence score
  - Per-day: aggregated 'sentiment_mean' = mean(pos_score - neg_score)

The model is loaded once and cached in the module-level variable
`_pipeline` to avoid repeated downloads during a Streamlit session.

Usage
-----
    from src.sentiment import score_headlines, aggregate_daily_sentiment

    headlines_df = ...   # DataFrame with 'date' and 'title' columns
    scored       = score_headlines(headlines_df)
    daily_sent   = aggregate_daily_sentiment(scored)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_pipeline = None  # module-level singleton


# ─────────────────────────── model loader ─────────────────────

def _get_pipeline():
    """Load FinBERT pipeline (once per process)."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline

        logger.info("Loading ProsusAI/finbert — first run may take a while…")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _pipeline = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,          # return all three label scores
            device=-1,           # CPU; set to 0 for GPU
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load FinBERT: %s", exc)
        _pipeline = None
    return _pipeline


# ─────────────────────────── scoring ──────────────────────────

def _extract_scores(raw_scores: list[dict]) -> dict:
    """
    Convert FinBERT's list-of-dicts output into a flat score dict.

    raw_scores example:
        [{'label': 'positive', 'score': 0.91},
         {'label': 'negative', 'score': 0.05},
         {'label': 'neutral',  'score': 0.04}]
    """
    scores = {item["label"].lower(): item["score"] for item in raw_scores}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    label = max(scores, key=scores.get)
    confidence = scores[label]
    net = pos - neg
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "score_positive": round(pos, 4),
        "score_negative": round(neg, 4),
        "score_neutral": round(neu, 4),
        "sentiment_net": round(net, 4),
    }


def score_headlines(
    headlines_df: pd.DataFrame,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Run FinBERT on each headline title and annotate the DataFrame.

    Parameters
    ----------
    headlines_df : DataFrame with at minimum columns ['date', 'title'].
    batch_size   : Number of headlines to process in one forward pass.

    Returns
    -------
    Original DataFrame with added columns:
      label, confidence, score_positive, score_negative,
      score_neutral, sentiment_net.

    If FinBERT fails to load, returns the original DataFrame with
    dummy zero sentiment columns and logs an error.
    """
    if headlines_df.empty or "title" not in headlines_df.columns:
        logger.warning("score_headlines: empty or missing 'title' column.")
        return _add_dummy_sentiment(headlines_df)

    pipe = _get_pipeline()
    if pipe is None:
        logger.error("FinBERT unavailable; returning zero-sentiment DataFrame.")
        return _add_dummy_sentiment(headlines_df)

    df = headlines_df.copy()
    titles = df["title"].fillna("").tolist()

    results: list[dict] = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        # Guard against empty strings
        batch = [t if t.strip() else "neutral news" for t in batch]
        try:
            raw = pipe(batch)
            # raw is a list of list-of-dicts when top_k=None
            for item in raw:
                results.append(_extract_scores(item))
        except Exception as exc:
            logger.warning("FinBERT batch %d failed: %s — using zeros.", i, exc)
            for _ in batch:
                results.append(
                    {"label": "neutral", "confidence": 0.0,
                     "score_positive": 0.0, "score_negative": 0.0,
                     "score_neutral": 1.0, "sentiment_net": 0.0}
                )

    scores_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, scores_df], axis=1)
    logger.info("Scored %d headlines with FinBERT.", len(df))
    return df


def _add_dummy_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with zeroed sentiment columns."""
    for col in ["label", "confidence", "score_positive",
                "score_negative", "score_neutral", "sentiment_net"]:
        df[col] = 0.0 if col != "label" else "neutral"
    return df


# ─────────────────────────── aggregation ──────────────────────

def aggregate_daily_sentiment(
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate headline-level FinBERT scores to daily scores.

    Parameters
    ----------
    scored_df : Output of score_headlines() with a 'date' column.

    Returns
    -------
    DataFrame indexed by date with columns:
      sentiment_mean  : mean net (pos-neg) score across the day's headlines
      sentiment_std   : std dev of net scores
      n_headlines     : count of headlines that day
      n_positive      : headlines labelled positive
      n_negative      : headlines labelled negative
      n_neutral       : headlines labelled neutral
    """
    if scored_df.empty or "sentiment_net" not in scored_df.columns:
        return pd.DataFrame(columns=["sentiment_mean"])

    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    daily = df.groupby("date").agg(
        sentiment_mean=("sentiment_net", "mean"),
        sentiment_std=("sentiment_net", "std"),
        n_headlines=("title", "count"),
        n_positive=("label", lambda s: (s == "positive").sum()),
        n_negative=("label", lambda s: (s == "negative").sum()),
        n_neutral=("label", lambda s: (s == "neutral").sum()),
    )
    daily.index = pd.to_datetime(daily.index)
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    logger.info("Daily sentiment aggregated for %d dates.", len(daily))
    return daily


# ─────────────────────────── convenience ──────────────────────

def build_sentiment_series(
    headlines: list[dict],
) -> Optional[pd.DataFrame]:
    """
    End-to-end helper: raw headline list → daily sentiment DataFrame.

    Returns None if headlines list is empty.
    """
    if not headlines:
        return None
    from src.data_loader import headlines_to_dataframe  # local import to avoid circular

    hdf = headlines_to_dataframe(headlines)
    if hdf.empty:
        return None

    scored = score_headlines(hdf)
    daily = aggregate_daily_sentiment(scored)
    return daily if not daily.empty else None
