"""
features.py
===========
Builds the feature matrix from raw OHLCV + daily sentiment scores.

Features computed:
  - Moving Averages:  MA_7, MA_21, MA_50
  - RSI(14)
  - Daily Return:     pct_change of Close
  - Rolling Volatility(21): std of daily returns
  - Lagged Close:     Close_lag1, Close_lag2, Close_lag3, Close_lag5
  - Lagged Returns:   Return_lag1, Return_lag2, Return_lag3, Return_lag5
  - Sentiment:        sentiment_mean (when available)
  - Target:           Next_Close (Close shifted -1; for regression)
  - Target_Direction: 1 if next close > today close else 0 (classification)

All scaler fitting MUST happen on the training slice only.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

logger = logging.getLogger(__name__)


# ─────────────────────────── technical indicators ─────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA_7, MA_21, MA_50 columns using the 'Close' column."""
    for window in (7, 21, 50):
        col = f"MA_{window}"
        df[col] = SMAIndicator(close=df["Close"], window=window).sma_indicator()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add RSI_{window} column."""
    df[f"RSI_{window}"] = RSIIndicator(close=df["Close"], window=window).rsi()
    return df


def add_returns_and_volatility(df: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """Add daily returns and rolling volatility."""
    df["Daily_Return"] = df["Close"].pct_change()
    df[f"Volatility_{vol_window}"] = (
        df["Daily_Return"].rolling(window=vol_window).std()
    )
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """Add lagged Close and Return features."""
    for lag in lags:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
        df[f"Return_lag{lag}"] = df["Daily_Return"].shift(lag)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create supervised-learning targets (no data leakage):
      - Next_Close      : tomorrow's closing price (regression target)
      - Target_Direction: 1 = price goes up, 0 = price goes down or flat
    """
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target_Direction"] = (df["Next_Close"] > df["Close"]).astype(int)
    return df


# ─────────────────────────── sentiment merge ──────────────────

def merge_sentiment(
    df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Left-join daily sentiment scores onto the price DataFrame.

    Parameters
    ----------
    df            : Price DataFrame with DatetimeIndex.
    sentiment_df  : DataFrame with DatetimeIndex and 'sentiment_mean' column,
                    or None / empty DataFrame when unavailable.

    Returns
    -------
    df with 'sentiment_mean' column (NaN-filled with 0 when not available).
    """
    if sentiment_df is not None and not sentiment_df.empty and "sentiment_mean" in sentiment_df.columns:
        sentiment_daily = sentiment_df[["sentiment_mean"]].copy()
        sentiment_daily.index = pd.to_datetime(sentiment_daily.index)
        df = df.join(sentiment_daily, how="left")
        filled = df["sentiment_mean"].isna().sum()
        if filled > 0:
            logger.debug("Filling %d missing sentiment values with 0.", filled)
        df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
        logger.info("Sentiment merged — %d trading days with scores.", (df["sentiment_mean"] != 0).sum())
    else:
        df["sentiment_mean"] = 0.0
        logger.info("No sentiment data available; 'sentiment_mean' set to 0.")
    return df


# ─────────────────────────── master pipeline ──────────────────

FEATURE_COLS = [
    "MA_7", "MA_21", "MA_50",
    "RSI_14",
    "Daily_Return", "Volatility_21",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5",
    "Return_lag1", "Return_lag2", "Return_lag3", "Return_lag5",
    "sentiment_mean",
    "Volume",
]

TARGET_COL = "Next_Close"
DIRECTION_COL = "Target_Direction"


def build_features(
    price_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    *,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    price_df     : Raw OHLCV DataFrame from data_loader.fetch_price_data().
    sentiment_df : Optional daily sentiment DataFrame from sentiment module.
    drop_na      : When True, drop rows that still have NaN after engineering.

    Returns
    -------
    DataFrame with all feature columns + targets, ready for train/test split.
    """
    df = price_df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_returns_and_volatility(df)
    df = add_lag_features(df)
    df = merge_sentiment(df, sentiment_df)
    df = add_target(df)

    if drop_na:
        before = len(df)
        df.dropna(inplace=True)
        logger.info("Dropped %d rows with NaN (feature warm-up). %d rows remain.", before - len(df), len(df))

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns that are present in *df* (some may be absent)."""
    return [c for c in FEATURE_COLS if c in df.columns]


def time_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict temporal train/test split — NO shuffling.

    Parameters
    ----------
    df         : Feature DataFrame with DatetimeIndex.
    test_ratio : Fraction of rows to use for testing.

    Returns
    -------
    (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info(
        "Train: %s → %s (%d rows) | Test: %s → %s (%d rows)",
        train.index[0].date(), train.index[-1].date(), len(train),
        test.index[0].date(), test.index[-1].date(), len(test),
    )
    return train, test
