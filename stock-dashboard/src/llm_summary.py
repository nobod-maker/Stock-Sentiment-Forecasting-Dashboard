"""
llm_summary.py
==============
Generates a 3-4 sentence plain-English explanation of recent stock
price action using an OpenAI-compatible LLM endpoint (llm7.io).

The prompt:
  - Provides last 5 days of OHLCV + technical indicators
  - Includes the daily sentiment summary and top headlines
  - Forbids fabrication beyond the given data
  - Appends the mandatory disclaimer: "This is not financial advice."
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.llm7.io/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "default")


# ─────────────────────────── client ───────────────────────────

def _get_client():
    """Return an openai.OpenAI client pointed at the configured base URL."""
    try:
        from openai import OpenAI
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    except ImportError as exc:
        raise ImportError("openai package not installed. Run: pip install openai") from exc


# ─────────────────────────── prompt builder ───────────────────

def _build_prompt(
    ticker: str,
    recent_prices: pd.DataFrame,
    latest_indicators: dict,
    sentiment_summary: dict,
    top_headlines: list[dict],
    next_day_forecast: Optional[float],
    selected_model: str,
) -> str:
    """
    Construct the LLM prompt from structured data.

    All inputs are numerical or text already retrieved from APIs —
    the LLM is explicitly instructed not to invent data.
    """
    # --- Price table (last 5 trading days) ---
    price_lines = []
    for dt, row in recent_prices.tail(5).iterrows():
        date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        price_lines.append(
            f"  {date_str}: Open={row.get('Open', 'N/A'):.2f}, "
            f"High={row.get('High', 'N/A'):.2f}, "
            f"Low={row.get('Low', 'N/A'):.2f}, "
            f"Close={row.get('Close', 'N/A'):.2f}, "
            f"Volume={int(row.get('Volume', 0)):,}"
        )
    price_table = "\n".join(price_lines)

    # --- Indicators ---
    ind_lines = []
    for k, v in latest_indicators.items():
        ind_lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    indicators_text = "\n".join(ind_lines) if ind_lines else "  No indicators available."

    # --- Sentiment ---
    sent_text = (
        f"  Overall net sentiment (pos–neg): {sentiment_summary.get('sentiment_mean', 0):.3f}\n"
        f"  Positive articles: {sentiment_summary.get('n_positive', 'N/A')}\n"
        f"  Negative articles: {sentiment_summary.get('n_negative', 'N/A')}\n"
        f"  Neutral  articles: {sentiment_summary.get('n_neutral',  'N/A')}"
    )

    # --- Headlines ---
    headline_lines = []
    for i, h in enumerate(top_headlines[:5], 1):
        label = h.get("label", "neutral").upper()
        headline_lines.append(f"  {i}. [{label}] {h.get('title', '')} ({h.get('date', '')})")
    headlines_text = "\n".join(headline_lines) if headline_lines else "  No headlines available."

    # --- Forecast ---
    forecast_text = (
        f"  Model: {selected_model}\n"
        f"  Predicted next-day closing price: ${next_day_forecast:.2f}"
        if next_day_forecast is not None
        else "  Forecast not available."
    )

    prompt = f"""You are a financial analyst assistant. Your task is to write a concise 3-4 sentence
plain-English explanation of why {ticker} moved the way it did recently, and what the forecast suggests.

IMPORTANT RULES:
- Base your explanation ONLY on the data provided below. Do not fabricate any facts, figures, or events.
- Do not mention any information that is not in the data below.
- Use professional but accessible language.
- End with exactly this sentence on its own line: "This is not financial advice."

--- DATA PROVIDED ---

LAST 5 TRADING DAYS (OHLCV):
{price_table}

LATEST TECHNICAL INDICATORS:
{indicators_text}

NEWS SENTIMENT SUMMARY (recent period):
{sent_text}

TOP HEADLINES:
{headlines_text}

FORECAST:
{forecast_text}

--- END OF DATA ---

Write your 3-4 sentence analysis below:"""

    return prompt


# ─────────────────────────── summary generator ────────────────

def generate_summary(
    ticker: str,
    recent_prices: pd.DataFrame,
    feature_df: pd.DataFrame,
    sentiment_summary: dict,
    top_headlines: list[dict],
    next_day_forecast: Optional[float] = None,
    selected_model: str = "XGBoost",
    max_tokens: int = 350,
    temperature: float = 0.4,
) -> str:
    """
    Call the LLM to generate an explanatory stock summary.

    Parameters
    ----------
    ticker            : Stock ticker (e.g. 'AAPL').
    recent_prices     : Raw OHLCV DataFrame (last N rows).
    feature_df        : Feature DataFrame (for extracting latest indicators).
    sentiment_summary : Dict of daily aggregated sentiment metrics.
    top_headlines     : List of headline dicts with sentiment labels.
    next_day_forecast : Optional predicted next-day price.
    selected_model    : Name of the model that produced the forecast.
    max_tokens        : Maximum LLM output tokens.
    temperature       : LLM sampling temperature (lower = more factual).

    Returns
    -------
    Plain-English summary string (always ends with disclaimer).
    On any API error, returns a safe fallback string.
    """
    if not LLM_API_KEY:
        return (
            "⚠️ LLM API key not configured. Set `LLM_API_KEY` in your `.env` file.\n\n"
            "_This is not financial advice._"
        )

    # Extract latest indicator values
    latest_row = feature_df.iloc[-1] if not feature_df.empty else pd.Series()
    indicator_keys = [
        "MA_7", "MA_21", "MA_50", "RSI_14",
        "Daily_Return", "Volatility_21", "sentiment_mean",
    ]
    latest_indicators = {
        k: float(latest_row[k])
        for k in indicator_keys
        if k in latest_row.index and pd.notna(latest_row[k])
    }

    prompt = _build_prompt(
        ticker=ticker,
        recent_prices=recent_prices,
        latest_indicators=latest_indicators,
        sentiment_summary=sentiment_summary,
        top_headlines=top_headlines,
        next_day_forecast=next_day_forecast,
        selected_model=selected_model,
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        summary = response.choices[0].message.content.strip()

        # Ensure disclaimer is present
        if "This is not financial advice." not in summary:
            summary += "\n\nThis is not financial advice."

        logger.info("LLM summary generated successfully (%d chars).", len(summary))
        return summary

    except Exception as exc:
        logger.error("LLM summary generation failed: %s", exc)
        return (
            f"⚠️ Unable to generate AI summary: {exc}\n\n"
            "This is not financial advice."
        )
