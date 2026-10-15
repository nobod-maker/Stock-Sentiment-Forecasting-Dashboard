"""
data_loader.py
==============
Fetches and caches:
  - OHLCV price data via yfinance
  - News headlines via NewsAPI (primary) → Finnhub (fallback)

All external calls are wrapped in try/except blocks with clear
logging. Results are serialised to local JSON cache files and
reused when the TTL has not expired.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────── config ───────────────────────────
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", 6))
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── cache helpers ────────────────────

def _cache_path(key: str) -> Path:
    """Return a deterministic file path for a cache key."""
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _load_cache(key: str) -> Optional[dict]:
    """Return cached payload if it exists and is within TTL."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        saved_at = datetime.fromisoformat(payload["_cached_at"])
        if datetime.utcnow() - saved_at < timedelta(hours=CACHE_TTL_HOURS):
            logger.debug("Cache hit: %s", key)
            return payload["data"]
        logger.debug("Cache expired: %s", key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache read error (%s): %s", key, exc)
    return None


def _save_cache(key: str, data: object) -> None:
    """Persist data to cache with current UTC timestamp."""
    path = _cache_path(key)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"_cached_at": datetime.utcnow().isoformat(), "data": data}, fh)
        logger.debug("Cache saved: %s", key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache write error (%s): %s", key, exc)


# ─────────────────────────── price data ───────────────────────

def fetch_price_data(
    ticker: str,
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker        : Yahoo Finance ticker symbol (e.g. 'AAPL').
    start         : Start date as 'YYYY-MM-DD'.
    end           : End date as 'YYYY-MM-DD'.
    force_refresh : Bypass local cache when True.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns
    [Open, High, Low, Close, Volume, Adj Close (if available)].
    Raises ValueError for invalid tickers or empty results.
    """
    cache_key = f"prices_{ticker}_{start}_{end}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            df = pd.read_json(io.StringIO(cached), orient="split")
            df.index = pd.to_datetime(df.index)
            return df

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed for '{ticker}': {exc}") from exc

    if raw is None or raw.empty:
        raise ValueError(
            f"No price data returned for ticker '{ticker}'. "
            "Check the symbol and date range."
        )

    # Flatten MultiIndex columns if present (yfinance ≥0.2.38 behaviour)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Forward-fill minor gaps (weekends already excluded by yfinance)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    _save_cache(cache_key, df.to_json(orient="split"))
    logger.info("Fetched %d rows of price data for %s", len(df), ticker)
    return df


# ─────────────────────────── news helpers ─────────────────────

def _newsapi_headlines(ticker: str, company_name: str, days: int = 30) -> list[dict]:
    """Fetch up to 100 recent headlines from NewsAPI."""
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set – skipping NewsAPI.")
        return []
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} OR {company_name}",
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "date": a.get("publishedAt", "")[:10],
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
            }
            for a in articles
            if a.get("title")
        ]
    except requests.exceptions.RequestException as exc:
        logger.warning("NewsAPI request failed: %s", exc)
        return []


def _finnhub_headlines(ticker: str, days: int = 365) -> list[dict]:
    """Fetch headlines from Finnhub company-news endpoint."""
    if not FINNHUB_KEY:
        logger.warning("FINNHUB_KEY not set – skipping Finnhub.")
        return []
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start_date,
        "to": end_date,
        "token": FINNHUB_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            logger.warning("Unexpected Finnhub response: %s", items)
            return []
        return [
            {
                "date": datetime.utcfromtimestamp(a.get("datetime", 0)).strftime(
                    "%Y-%m-%d"
                ),
                "title": a.get("headline", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
            }
            for a in items
            if a.get("headline")
        ]
    except requests.exceptions.RequestException as exc:
        logger.warning("Finnhub request failed: %s", exc)
        return []


def _get_company_name(ticker: str) -> str:
    """Best-effort lookup of long name via yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:  # noqa: BLE001
        return ticker


def fetch_news_headlines(
    ticker: str,
    *,
    days: int = 30,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch recent news headlines for *ticker*.

    Strategy:
      1. Check local cache.
      2. Try NewsAPI (last ~30 days).
      3. Fall back to Finnhub (up to 1 year).
      4. Return empty list if both fail, logging a warning.

    Each headline dict has keys: date, title, source, url.
    """
    cache_key = f"news_{ticker}_{days}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    company = _get_company_name(ticker)
    headlines = _newsapi_headlines(ticker, company, days=days)

    if not headlines:
        logger.info("NewsAPI returned nothing; trying Finnhub…")
        headlines = _finnhub_headlines(ticker, days=max(days, 365))

    if not headlines:
        logger.warning(
            "No headlines found for '%s'. Sentiment will not be available.", ticker
        )

    # De-duplicate by title
    seen: set[str] = set()
    unique: list[dict] = []
    for h in headlines:
        if h["title"] not in seen:
            seen.add(h["title"])
            unique.append(h)

    # Sort newest first
    unique.sort(key=lambda x: x["date"], reverse=True)
    _save_cache(cache_key, unique)
    logger.info("Fetched %d unique headlines for %s", len(unique), ticker)
    return unique


def headlines_to_dataframe(headlines: list[dict]) -> pd.DataFrame:
    """
    Convert raw headline list to a tidy DataFrame with a DatetimeIndex.

    Returns an empty DataFrame (not None) when the input list is empty.
    """
    if not headlines:
        return pd.DataFrame(columns=["date", "title", "source", "url"])
    df = pd.DataFrame(headlines)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
