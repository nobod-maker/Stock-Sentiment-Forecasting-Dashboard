"""
app.py
======
AURA QUANT // TERMINAL
Institutional Stock Sentiment & Machine Learning Forecasting Engine
=====================================================================
Streamlit entry-point.
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── ensure src/ is importable when running from project root ──
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from src.data_loader import fetch_price_data, fetch_news_headlines, headlines_to_dataframe
from src.features import build_features, get_feature_columns, time_split, FEATURE_COLS
from src.models import fit_naive, fit_linear, fit_xgboost, predict_next_day, FitResult
from src.evaluate import (
    build_metrics_table,
    metrics_bar_chart,
    directional_accuracy_chart,
    forecast_vs_actual_chart,
    feature_importance_chart,
)
from src.sentiment import score_headlines, aggregate_daily_sentiment, build_sentiment_series
from src.llm_summary import generate_summary

# ─────────────────────────── logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("dashboard")


# ─────────────────────────── page config ──────────────────────
st.set_page_config(
    page_title="AURA QUANT // Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── modern design system CSS ─────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #F8FAFC;
    }

    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1480px;
    }

    /* Pulse animation for live data status */
    @keyframes live-pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        animation: live-pulse 2s infinite;
    }

    /* Top Brand Navigation Header */
    .brand-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        background: linear-gradient(90deg, rgba(17, 24, 39, 0.7) 0%, rgba(15, 23, 42, 0.4) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
    }

    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon-box {
        background: linear-gradient(135deg, #0284C7 0%, #38BDF8 50%, #818CF8 100%);
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35);
    }

    .brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-sub {
        font-size: 0.68rem;
        font-weight: 600;
        color: #64748B;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .live-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #34D399;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Modern KPI Strip */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }

    .kpi-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 18px;
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-1px);
    }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }

    .kpi-sub {
        font-size: 0.75rem;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Badges & Pills */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .badge-positive {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-negative {
        background: rgba(244, 63, 94, 0.14);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .badge-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }

    .badge-cyan {
        background: rgba(56, 189, 248, 0.14);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Streamlit Default Metric Styling Override */
    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 12px 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tabs Bar Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 16px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.88rem;
        color: #94A3B8;
        background-color: transparent;
        border: none;
        transition: all 0.15s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background-color: rgba(255, 255, 255, 0.04);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.12) !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
        border: 1px solid rgba(56, 189, 248, 0.28) !important;
    }

    /* Prediction Hero Card */
    .forecast-hero-card {
        background: radial-gradient(100% 100% at 50% 0%, #151F33 0%, #0B101D 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 30px 24px;
        text-align: center;
        margin: 16px auto;
        max-width: 740px;
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.7), 0 0 25px rgba(56, 189, 248, 0.08);
        position: relative;
    }

    .forecast-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .forecast-price-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.2rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 6px 0;
    }

    .forecast-range-box {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 6px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #CBD5E1;
        margin-top: 8px;
    }

    .forecast-meta-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 20px;
        padding-top: 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }

    .forecast-meta-item-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .forecast-meta-item-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-top: 2px;
    }

    .legal-disclaimer {
        color: #64748B;
        font-size: 0.72rem;
        line-height: 1.5;
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px dashed rgba(255, 255, 255, 0.06);
    }

    /* News Feed Item */
    .news-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }

    .news-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-1px);
        background: #131E33;
    }

    .news-title {
        color: #E2E8F0;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.45;
        text-decoration: none;
        display: block;
        margin: 6px 0;
    }

    .news-title:hover {
        color: #38BDF8;
    }

    .news-meta {
        color: #64748B;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* AI Executive Briefing Card */
    .ai-briefing-card {
        background: linear-gradient(180deg, #101827 0%, #0D1424 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 5px solid #38BDF8;
        border-radius: 14px;
        padding: 24px 28px;
        margin: 16px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    .ai-briefing-title {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .ai-briefing-content {
        font-size: 1.02rem;
        line-height: 1.8;
        color: #E2E8F0;
    }

    /* Quick Ticker Chips */
    .ticker-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 6px;
        margin-bottom: 14px;
    }

    /* Primary Action Button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.95rem;
        border-radius: 9px;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.4);
        transition: all 0.2s ease;
    }

    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
        box-shadow: 0 6px 22px rgba(14, 165, 233, 0.55);
        transform: translateY(-1px);
    }

    /* Technical Indicator Pill Ribbon */
    .tech-pill-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin: 14px 0;
    }

    .tech-pill {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 14px;
    }

    .tech-pill-label {
        font-size: 0.7rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    .tech-pill-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# CACHED DATA PIPELINE
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="📥 Downloading market price history…")
def cached_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and cache OHLCV price data."""
    return fetch_price_data(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner="📰 Querying news wires & feeds…")
def cached_headlines(ticker: str, days: int = 60) -> list[dict]:
    """Fetch and cache news headlines."""
    return fetch_news_headlines(ticker, days=days)


@st.cache_data(ttl=7200, show_spinner="🤖 Scoring headlines with ProsusAI/FinBERT…")
def cached_sentiment(headlines_json: str) -> tuple[list[dict], Optional[pd.DataFrame]]:
    """
    Score headlines with FinBERT and aggregate to daily scores.
    Input is JSON string for hash stability in cache.
    """
    import json
    headlines = json.loads(headlines_json)
    if not headlines:
        return [], None

    hdf = headlines_to_dataframe(headlines)
    if hdf.empty:
        return [], None

    scored_df = score_headlines(hdf)
    daily_df = aggregate_daily_sentiment(scored_df)
    scored_list = scored_df.to_dict(orient="records")
    return scored_list, daily_df if not daily_df.empty else None


@st.cache_data(ttl=1800, show_spinner="⚙️ Engineering quant features & training models…")
def cached_pipeline(
    ticker: str,
    start: str,
    end: str,
    has_sentiment: bool,
    selected_models: list[str],
) -> dict:
    """
    Full ML & technical analysis pipeline cached to avoid retraining on layout toggles.
    """
    price_df = fetch_price_data(ticker, start, end)
    headlines = fetch_news_headlines(ticker, days=60)

    # Sentiment processing
    import json
    scored_list, daily_sent = cached_sentiment(json.dumps(headlines))
    scored_df = pd.DataFrame(scored_list) if scored_list else pd.DataFrame()

    feature_df = build_features(price_df, daily_sent)
    feat_cols = get_feature_columns(feature_df)
    train_df, test_df = time_split(feature_df, test_ratio=0.2)

    results: dict[str, FitResult] = {}

    # Always fit Naive baseline for comparison
    results["Naive Baseline"] = fit_naive(train_df, test_df)

    if "Linear Regression" in selected_models:
        results["Linear Regression"] = fit_linear(train_df, test_df, feat_cols)

    if "XGBoost" in selected_models:
        results["XGBoost"] = fit_xgboost(train_df, test_df, feat_cols)

    if "LSTM (PyTorch)" in selected_models:
        try:
            from src.models import fit_lstm
            results["LSTM (PyTorch)"] = fit_lstm(train_df, test_df, feat_cols)
        except Exception as e:
            logger.warning("LSTM training skipped: %s", e)

    return {
        "price_df": price_df,
        "feature_df": feature_df,
        "train_df": train_df,
        "test_df": test_df,
        "feat_cols": feat_cols,
        "results": results,
        "daily_sent": daily_sent,
        "scored_df": scored_df,
        "headlines": headlines,
    }


# ═══════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render modern sidebar with quick presets and parameter controls."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
                <div style="background:linear-gradient(135deg, #0284C7, #38BDF8); width:36px; height:36px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-size:1.15rem; box-shadow:0 4px 14px rgba(2,132,199,0.4);">
                    ⚡
                </div>
                <div>
                    <div style="font-weight:800; font-size:1.1rem; color:#F8FAFC; letter-spacing:-0.01em;">AURA QUANT</div>
                    <div style="font-size:0.68rem; color:#64748B; font-weight:600; letter-spacing:0.06em; text-transform:uppercase;">Terminal Config</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:0.72rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Market Asset (Ticker)</div>',
            unsafe_allow_html=True,
        )

        default_ticker = st.session_state.get("active_ticker", "AAPL")
        ticker = st.text_input(
            "Ticker Symbol",
            value=default_ticker,
            max_chars=10,
            label_visibility="collapsed",
            help="Enter a valid Yahoo Finance ticker (e.g. AAPL, NVDA, TSLA, MSFT)",
        ).upper().strip()

        # Popular Tickers Quick Bar
        st.markdown(
            '<div style="font-size:0.68rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.05em; margin-top:4px; margin-bottom:4px;">Quick Select:</div>',
            unsafe_allow_html=True,
        )
        pop_cols = st.columns(4)
        popular_tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "SPY"]
        for idx, sym in enumerate(popular_tickers):
            col_target = pop_cols[idx % 4]
            if col_target.button(sym, key=f"btn_pop_{sym}", use_container_width=True):
                st.session_state["active_ticker"] = sym
                ticker = sym
                st.rerun()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.72rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Historical Lookback</div>',
            unsafe_allow_html=True,
        )

        # Date Range Presets
        preset_cols = st.columns(4)
        today = date.today()
        if "start_date" not in st.session_state:
            st.session_state["start_date"] = today - timedelta(days=3 * 365)
        if "end_date" not in st.session_state:
            st.session_state["end_date"] = today

        if preset_cols[0].button("1Y", use_container_width=True):
            st.session_state["start_date"] = today - timedelta(days=365)
            st.session_state["end_date"] = today
            st.rerun()
        if preset_cols[1].button("2Y", use_container_width=True):
            st.session_state["start_date"] = today - timedelta(days=2 * 365)
            st.session_state["end_date"] = today
            st.rerun()
        if preset_cols[2].button("3Y", use_container_width=True):
            st.session_state["start_date"] = today - timedelta(days=3 * 365)
            st.session_state["end_date"] = today
            st.rerun()
        if preset_cols[3].button("5Y", use_container_width=True):
            st.session_state["start_date"] = today - timedelta(days=5 * 365)
            st.session_state["end_date"] = today
            st.rerun()

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("From", value=st.session_state["start_date"])
        with col_d2:
            end_date = st.date_input("To", value=st.session_state["end_date"])

        st.markdown(
            '<div style="font-size:0.72rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-top:14px; margin-bottom:6px;">Model Selection</div>',
            unsafe_allow_html=True,
        )
        available_models = ["Linear Regression", "XGBoost", "LSTM (PyTorch)"]
        selected_models = st.multiselect(
            "Select models",
            options=available_models,
            default=["Linear Regression", "XGBoost"],
            label_visibility="collapsed",
        )
        selected_models_full = ["Naive Baseline"] + selected_models

        st.markdown(
            '<div style="font-size:0.72rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-top:14px; margin-bottom:6px;">Primary Forecast Engine</div>',
            unsafe_allow_html=True,
        )
        forecast_model = st.selectbox(
            "Primary Model",
            options=selected_models_full,
            index=min(2, len(selected_models_full) - 1),
            label_visibility="collapsed",
        )

        with st.expander("🛠️ Chart & Display Controls", expanded=False):
            show_mas = st.checkbox("Show Moving Averages (7, 21, 50)", value=True)
            show_bb = st.checkbox("Show Bollinger Bands (20, 2)", value=True)
            show_vol = st.checkbox("Show Trading Volume Panel", value=True)
            show_rsi = st.checkbox("Show RSI(14) Panel", value=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚡ Run Quant Analysis", type="primary", use_container_width=True)

        st.markdown(
            """
            <div style="margin-top:22px; padding:12px 14px; background:#0B0F19; border:1px solid rgba(255,255,255,0.06); border-radius:10px; font-size:0.72rem; color:#64748B; line-height:1.6;">
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px;">
                    <span class="pulse-dot"></span>
                    <span style="color:#94A3B8; font-weight:600;">Feed: Yahoo Finance + Finnhub</span>
                </div>
                <div>NLP: ProsusAI/FinBERT (HuggingFace)</div>
                <div>Engine: scikit-learn &amp; XGBoost</div>
                <div>LLM: OpenAI API Compatible</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "ticker": ticker,
        "start": str(start_date),
        "end": str(end_date),
        "selected_models": selected_models_full,
        "forecast_model": forecast_model,
        "show_mas": show_mas,
        "show_bb": show_bb,
        "show_vol": show_vol,
        "show_rsi": show_rsi,
        "run": run_btn,
    }


# ═══════════════════════════════════════════════════════════════
# CHART ENGINE (PLOTLY DARK THEME)
# ═══════════════════════════════════════════════════════════════

def candlestick_chart(
    price_df: pd.DataFrame,
    ticker: str,
    show_mas: bool = True,
    show_bb: bool = True,
) -> go.Figure:
    """Candlestick chart with MA overlays, Bollinger Bands, and interactive range buttons."""
    from ta.volatility import BollingerBands

    fig = go.Figure()

    # Bollinger Bands
    if show_bb and len(price_df) >= 20:
        bb = BollingerBands(close=price_df["Close"], window=20, window_dev=2)
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()

        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=bb_high,
                mode="lines",
                name="BB Upper (20,2)",
                line=dict(color="rgba(148, 163, 184, 0.35)", width=1, dash="dot"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=bb_low,
                mode="lines",
                name="BB Lower (20,2)",
                line=dict(color="rgba(148, 163, 184, 0.35)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.04)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=bb_mid,
                mode="lines",
                name="BB Mid (20)",
                line=dict(color="rgba(148, 163, 184, 0.45)", width=1, dash="dash"),
            )
        )

    # OHLCV Candlestick
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name="OHLCV",
            increasing_line_color="#10B981",
            increasing_fillcolor="#10B981",
            decreasing_line_color="#F43F5E",
            decreasing_fillcolor="#F43F5E",
        )
    )

    # Moving Averages
    if show_mas:
        ma_configs = [
            ("MA_7", "#38BDF8", "MA 7"),
            ("MA_21", "#818CF8", "MA 21"),
            ("MA_50", "#FBBF24", "MA 50"),
        ]
        for col_name, color, label in ma_configs:
            if col_name in price_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_df.index,
                        y=price_df[col_name],
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=1.6),
                    )
                )

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> — Price History & Volatility Regimes",
            font=dict(size=15, color="#F8FAFC", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94A3B8", size=11),
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]),
                bgcolor="#111827",
                activecolor="#0284C7",
                font=dict(color="#94A3B8", size=10),
            ),
        ),
        yaxis=dict(
            title="Close Price (USD)",
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94A3B8", size=11),
            tickprefix="$",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=15, r=15, t=55, b=15),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1", size=11),
        ),
    )
    return fig


def volume_chart(price_df: pd.DataFrame) -> go.Figure:
    """Volume bar chart with 20-period moving average overlay."""
    colors = [
        "rgba(16, 185, 129, 0.65)" if c >= o else "rgba(244, 63, 94, 0.65)"
        for c, o in zip(price_df["Close"], price_df["Open"])
    ]
    vol_ma = price_df["Volume"].rolling(window=20).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=price_df.index,
            y=price_df["Volume"],
            marker_color=colors,
            marker_line_width=0,
            name="Volume",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=vol_ma,
            mode="lines",
            name="20-Day Vol MA",
            line=dict(color="#FBBF24", width=1.5),
        )
    )
    fig.update_layout(
        title=dict(
            text="Trading Volume with 20D Moving Average",
            font=dict(size=13, color="#94A3B8", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#64748B", size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#64748B", size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=15, r=15, t=40, b=15),
        showlegend=False,
    )
    return fig


def rsi_chart(feature_df: pd.DataFrame) -> go.Figure:
    """RSI indicator with gradient fill, overbought (70) and oversold (30) bands."""
    fig = go.Figure()
    if "RSI_14" in feature_df.columns:
        fig.add_trace(
            go.Scatter(
                x=feature_df.index,
                y=feature_df["RSI_14"],
                mode="lines",
                name="RSI(14)",
                line=dict(color="#A78BFA", width=1.8),
                fill="tozeroy",
                fillcolor="rgba(167, 139, 250, 0.06)",
            )
        )
    fig.add_hrect(
        y0=70, y1=100, fillcolor="rgba(244, 63, 94, 0.12)", line_width=0,
        annotation_text="Overbought 70", annotation_position="top right",
        annotation_font=dict(size=10, color="#FB7185")
    )
    fig.add_hrect(
        y0=0, y1=30, fillcolor="rgba(16, 185, 129, 0.12)", line_width=0,
        annotation_text="Oversold 30", annotation_position="bottom right",
        annotation_font=dict(size=10, color="#34D399")
    )
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(148, 163, 184, 0.3)", line_width=1)
    fig.update_layout(
        title=dict(
            text="Relative Strength Index (RSI 14)",
            font=dict(size=13, color="#94A3B8", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#64748B", size=10),
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#64748B", size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=15, r=15, t=40, b=15),
        showlegend=False,
    )
    return fig


def render_sentiment_chart(daily_sent: pd.DataFrame) -> None:
    """Bar chart of daily sentiment scores with clean styling."""
    if daily_sent is None or daily_sent.empty:
        st.info("Daily sentiment series unavailable.")
        return

    colors = [
        "#10B981" if v >= 0 else "#F43F5E" for v in daily_sent["sentiment_mean"]
    ]
    fig = go.Figure(
        go.Bar(
            x=daily_sent.index,
            y=daily_sent["sentiment_mean"],
            marker_color=colors,
            marker_line_width=0,
            name="Net Sentiment",
        )
    )
    fig.add_hline(y=0, line_color="rgba(148, 163, 184, 0.4)", line_width=1)
    fig.update_layout(
        title=dict(
            text="FinBERT Daily Net Sentiment Momentum (Pos − Neg)",
            font=dict(size=13, color="#94A3B8", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#64748B", size=10),
        ),
        yaxis=dict(
            range=[-1.05, 1.05],
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#64748B", size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=15, r=15, t=40, b=15),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    opts = render_sidebar()
    ticker = opts["ticker"]

    # Modern Top Navigation / Branding Strip
    st.markdown(
        f"""
        <div class="brand-nav">
            <div class="brand-title-wrap">
                <div class="brand-icon-box">⚡</div>
                <div>
                    <div class="brand-name">AURA QUANT // TERMINAL</div>
                    <div class="brand-sub">Machine Learning Price Forecasting &amp; FinBERT News NLP</div>
                </div>
            </div>
            <div class="live-status-pill">
                <span class="pulse-dot"></span>
                <span>ASSET: {ticker}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Auto-run on launch or explicit click ───────────────────
    should_run = opts["run"] or ("pipeline_data" not in st.session_state) or (st.session_state.get("current_ticker") != ticker)

    if should_run:
        with st.spinner(f"⚡ Ingesting {ticker} market streams & computing predictive features…"):
            try:
                data = cached_pipeline(
                    ticker=ticker,
                    start=opts["start"],
                    end=opts["end"],
                    has_sentiment=True,
                    selected_models=opts["selected_models"],
                )
                st.session_state["pipeline_data"] = data
                st.session_state["current_ticker"] = ticker
            except Exception as exc:
                st.error(f"❌ Market Ingestion Error for {ticker}: {exc}")
                logger.exception("Pipeline execution failed")
                return

    data = st.session_state["pipeline_data"]
    price_df = data["price_df"]
    feature_df = data["feature_df"]
    results: dict[str, FitResult] = data["results"]
    daily_sent: Optional[pd.DataFrame] = data["daily_sent"]
    scored_df: pd.DataFrame = data["scored_df"]
    headlines: list[dict] = data["headlines"]

    # ── Top Real-Time KPI Strip ───────────────────────────────
    latest = price_df.iloc[-1]
    prev = price_df.iloc[-2] if len(price_df) > 1 else latest
    price_change = latest["Close"] - prev["Close"]
    pct_change = (price_change / prev["Close"]) * 100
    chg_sign = "+" if price_change >= 0 else ""
    chg_color = "#34D399" if price_change >= 0 else "#FB7185"
    chg_badge = "badge-positive" if price_change >= 0 else "badge-negative"
    chg_arrow = "▲" if price_change >= 0 else "▼"

    # 52-week stats
    last_year = price_df.tail(252) if len(price_df) >= 252 else price_df
    high_52w = last_year["High"].max()
    low_52w = last_year["Low"].min()

    # RSI condition
    rsi_val = feature_df.iloc[-1].get("RSI_14", 50.0) if not feature_df.empty else 50.0
    if rsi_val >= 70:
        rsi_label = "OVERBOUGHT"
        rsi_badge = "badge-negative"
    elif rsi_val <= 30:
        rsi_label = "OVERSOLD"
        rsi_badge = "badge-positive"
    else:
        rsi_label = "NEUTRAL"
        rsi_badge = "badge-neutral"

    # Sentiment net
    has_sentiment = daily_sent is not None and not daily_sent.empty
    if has_sentiment:
        latest_sent_val = float(daily_sent.iloc[-1]["sentiment_mean"])
        sent_badge = "badge-positive" if latest_sent_val > 0.05 else ("badge-negative" if latest_sent_val < -0.05 else "badge-neutral")
        sent_text = "BULLISH" if latest_sent_val > 0.05 else ("BEARISH" if latest_sent_val < -0.05 else "NEUTRAL")
    else:
        latest_sent_val = 0.0
        sent_badge = "badge-neutral"
        sent_text = "NEUTRAL"

    # Render KPI Strip
    st.markdown(
        f"""
        <div class="kpi-strip">
            <div class="kpi-card">
                <div class="kpi-label">
                    <span>Spot Close</span>
                    <span class="badge {chg_badge}">{chg_arrow} {chg_sign}{pct_change:.2f}%</span>
                </div>
                <div class="kpi-value">${latest['Close']:.2f}</div>
                <div class="kpi-sub" style="color:{chg_color};">
                    <span>{chg_sign}${price_change:.2f} today</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">
                    <span>Day Range</span>
                    <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#94A3B8;">${latest['Low']:.2f} - ${latest['High']:.2f}</span>
                </div>
                <div class="kpi-value" style="font-size:1.35rem;">
                    ${latest['High'] - latest['Low']:.2f} <span style="font-size:0.8rem; color:#64748B;">spread</span>
                </div>
                <div class="kpi-sub" style="color:#94A3B8;">
                    Open: <strong style="color:#CBD5E1;">${latest['Open']:.2f}</strong>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">
                    <span>52-Week Range</span>
                    <span class="badge badge-neutral">1 Year</span>
                </div>
                <div class="kpi-value" style="font-size:1.35rem;">
                    ${low_52w:.2f} <span style="font-size:0.8rem; color:#64748B;">to</span> ${high_52w:.2f}
                </div>
                <div class="kpi-sub" style="color:#94A3B8;">
                    Volume: <strong style="color:#CBD5E1;">{int(latest['Volume']):,}</strong>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">
                    <span>RSI Momentum</span>
                    <span class="badge {rsi_badge}">{rsi_label}</span>
                </div>
                <div class="kpi-value" style="color:#A78BFA;">{rsi_val:.1f}</div>
                <div class="kpi-sub" style="color:#94A3B8;">
                    14-Period Strength Index
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">
                    <span>FinBERT Sentiment</span>
                    <span class="badge {sent_badge}">{sent_text}</span>
                </div>
                <div class="kpi-value" style="color:{chg_color};">{latest_sent_val:+.3f}</div>
                <div class="kpi-sub" style="color:#94A3B8;">
                    Net News Wire Score
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════
    # TABS INTERFACE
    # ═══════════════════════════════════════════════════════════
    tabs = st.tabs([
        "📊 Market & Technicals",
        "🔮 ML Forecast Engine",
        "📰 FinBERT Intelligence",
        "🤖 AI Strategist",
        "📋 Quant Benchmark",
    ])

    # ──────────────────────────────────────────────────────────
    # TAB 1: MARKET & TECHNICALS
    # ──────────────────────────────────────────────────────────
    with tabs[0]:
        # Merge MAs into price_df copy for chart rendering
        chart_df = price_df.copy()
        for col in ["MA_7", "MA_21", "MA_50"]:
            if col in feature_df.columns:
                chart_df[col] = feature_df[col]

        st.plotly_chart(
            candlestick_chart(
                chart_df,
                ticker=ticker,
                show_mas=opts["show_mas"],
                show_bb=opts["show_bb"],
            ),
            use_container_width=True,
        )

        col_vol, col_rsi = st.columns(2)
        with col_vol:
            if opts["show_vol"]:
                st.plotly_chart(volume_chart(price_df), use_container_width=True)
        with col_rsi:
            if opts["show_rsi"]:
                st.plotly_chart(rsi_chart(feature_df), use_container_width=True)

        # Technical Condition Summary Matrix
        ma50_val = feature_df.iloc[-1].get("MA_50", None)
        trend_regime = "Bullish (Above 50-SMA)" if (ma50_val and latest["Close"] > ma50_val) else "Bearish (Below 50-SMA)"
        vol_21 = feature_df.iloc[-1].get("Volatility_21", 0.0) * np.sqrt(252) * 100

        st.markdown(
            f"""
            <div class="tech-pill-grid">
                <div class="tech-pill">
                    <div class="tech-pill-label">Trend Regime</div>
                    <div class="tech-pill-val" style="font-size:0.95rem; color:#38BDF8;">{trend_regime}</div>
                </div>
                <div class="tech-pill">
                    <div class="tech-pill-label">21D Annualized Volatility</div>
                    <div class="tech-pill-val">{vol_21:.2f}%</div>
                </div>
                <div class="tech-pill">
                    <div class="tech-pill-label">7-Day Moving Average</div>
                    <div class="tech-pill-val">${feature_df.iloc[-1].get('MA_7', latest['Close']):.2f}</div>
                </div>
                <div class="tech-pill">
                    <div class="tech-pill-label">21-Day Moving Average</div>
                    <div class="tech-pill-val">${feature_df.iloc[-1].get('MA_21', latest['Close']):.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────────────────────
    # TAB 2: ML FORECAST ENGINE
    # ──────────────────────────────────────────────────────────
    with tabs[1]:
        primary_model = opts["forecast_model"]
        if primary_model in results:
            result = results[primary_model]
            st.plotly_chart(
                forecast_vs_actual_chart(result, price_df),
                use_container_width=True,
            )
        else:
            st.info(f"Model '{primary_model}' is not trained. Check sidebar selections.")

        # Best Model Prediction Hero Card
        best_model_name = (
            "XGBoost"
            if "XGBoost" in results
            else ("Linear Regression" if "Linear Regression" in results else "Naive Baseline")
        )
        best_result = results.get(best_model_name)
        if best_result is not None:
            latest_row = feature_df.iloc[-1]
            try:
                next_price = predict_next_day(best_result, latest_row)
                current_close = float(latest["Close"])
                price_delta = next_price - current_close
                pct_delta = (price_delta / current_close) * 100

                direction = "BULLISH" if price_delta >= 0 else "BEARISH"
                arrow = "▲" if price_delta >= 0 else "▼"
                pill_bg = "rgba(16, 185, 129, 0.14)" if price_delta >= 0 else "rgba(244, 63, 94, 0.14)"
                pill_color = "#34D399" if price_delta >= 0 else "#FB7185"
                pill_border = "rgba(16, 185, 129, 0.3)" if price_delta >= 0 else "rgba(244, 63, 94, 0.3)"
                delta_sign = "+" if price_delta >= 0 else ""

                # Expected 95% confidence corridor based on 21D volatility
                daily_vol = float(latest_row.get("Volatility_21", 0.015))
                lower_band = next_price * (1.0 - 1.96 * daily_vol)
                upper_band = next_price * (1.0 + 1.96 * daily_vol)

                st.markdown(
                    f"""
                    <div class="forecast-hero-card">
                        <div class="forecast-title">T+1 Predictive Machine Learning Forecast</div>
                        <div style="margin-bottom:8px;">
                            <span class="badge" style="background:{pill_bg}; color:{pill_color}; border:1px solid {pill_border}; font-size:0.85rem; padding:5px 14px;">
                                {arrow} {direction} TRAJECTORY ({delta_sign}{pct_delta:.2f}%)
                            </span>
                        </div>
                        <div class="forecast-price-display">${next_price:.2f}</div>
                        <div style="font-family:'JetBrains Mono'; color:{pill_color}; font-size:1.15rem; font-weight:600;">
                            {delta_sign}${price_delta:.2f} vs Spot Close (${current_close:.2f})
                        </div>
                        <div class="forecast-range-box">
                            <span>95% Volatility Corridor:</span>
                            <strong style="color:#38BDF8;">${lower_band:.2f}</strong> — <strong style="color:#38BDF8;">${upper_band:.2f}</strong>
                        </div>
                        <div class="forecast-meta-grid">
                            <div>
                                <div class="forecast-meta-item-label">Primary Estimator</div>
                                <div class="forecast-meta-item-val" style="color:#38BDF8;">{best_model_name}</div>
                            </div>
                            <div>
                                <div class="forecast-meta-item-label">Horizon</div>
                                <div class="forecast-meta-item-val">Next Session (T+1)</div>
                            </div>
                            <div>
                                <div class="forecast-meta-item-label">Test Split RMSE</div>
                                <div class="forecast-meta-item-val">${np.sqrt(np.mean((best_result.test_actuals - best_result.test_preds)**2)):.3f}</div>
                            </div>
                        </div>
                        <div class="legal-disclaimer">
                            ⚠️ Algorithmic quantitative estimate generated via gradient boosting and FinBERT sentiment features. Financial securities involve risk of capital loss. <strong>This terminal does not provide financial advice.</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Prediction Calculation Error: {exc}")

        # Multi-Model Consensus Matrix
        st.markdown(
            '<div style="font-size:0.88rem; font-weight:700; color:#CBD5E1; text-transform:uppercase; letter-spacing:0.04em; margin-top:20px; margin-bottom:10px;">Ensemble Forecast Comparison</div>',
            unsafe_allow_html=True,
        )
        consensus_rows = []
        for m_name, m_res in results.items():
            try:
                p_next = predict_next_day(m_res, feature_df.iloc[-1])
                diff = p_next - latest["Close"]
                p_chg = (diff / latest["Close"]) * 100
                test_rmse = np.sqrt(np.mean((m_res.test_actuals - m_res.test_preds)**2))
                consensus_rows.append({
                    "Model": m_name,
                    "Predicted Close": f"${p_next:.2f}",
                    "Expected Shift": f"{'+' if diff>=0 else ''}${diff:.2f}",
                    "Return (%)": f"{'+' if p_chg>=0 else ''}{p_chg:.2f}%",
                    "Direction": "BULLISH" if diff >= 0 else "BEARISH",
                    "Test RMSE": round(test_rmse, 4),
                })
            except Exception:
                continue

        if consensus_rows:
            cdf = pd.DataFrame(consensus_rows)
            st.dataframe(cdf, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────
    # TAB 3: FINBERT INTELLIGENCE
    # ──────────────────────────────────────────────────────────
    with tabs[2]:
        col_feed, col_stats = st.columns([1.6, 1])

        with col_feed:
            st.markdown(
                '<div style="font-size:0.88rem; font-weight:700; color:#CBD5E1; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:10px;">Financial Wire Sentiment Stream</div>',
                unsafe_allow_html=True,
            )

            # Sentiment Filter
            f_col1, f_col2 = st.columns([1, 1.5])
            with f_col1:
                filter_choice = st.selectbox("Filter Sentiment", ["All Articles", "Positive", "Negative", "Neutral"], label_visibility="collapsed")
            with f_col2:
                search_q = st.text_input("Search headlines", placeholder="Filter by keyword...", label_visibility="collapsed").lower().strip()

            if scored_df.empty:
                st.info("No headlines currently available.")
            else:
                display_df = scored_df.copy()
                if filter_choice != "All Articles":
                    display_df = display_df[display_df["label"].str.lower() == filter_choice.lower()]
                if search_q:
                    display_df = display_df[display_df["title"].str.lower().str.contains(search_q, na=False)]

                if "date" in display_df.columns:
                    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%b %d, %Y")

                for _, row in display_df.head(15).iterrows():
                    lbl = str(row.get("label", "neutral")).lower()
                    b_class = f"badge-{lbl}"
                    conf = row.get("confidence", 0.0)
                    title = row.get("title", "")
                    src = row.get("source", "Market News")
                    dt = row.get("date", "")
                    url = row.get("url", "#")
                    arrow_icon = "▲" if lbl == "positive" else ("▼" if lbl == "negative" else "●")

                    st.markdown(
                        f"""
                        <div class="news-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="badge {b_class}">{arrow_icon} {lbl.upper()} {conf:.0%}</span>
                                <span class="news-meta">{dt}</span>
                            </div>
                            <a href="{url}" target="_blank" class="news-title">{title} ↗</a>
                            <div class="news-meta">Publisher: <span style="color:#94A3B8;">{src}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with col_stats:
            st.markdown(
                '<div style="font-size:0.88rem; font-weight:700; color:#CBD5E1; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:10px;">Sentiment Distribution</div>',
                unsafe_allow_html=True,
            )
            render_sentiment_chart(daily_sent)

            if not scored_df.empty and "label" in scored_df.columns:
                counts = scored_df["label"].str.lower().value_counts()
                pos_cnt = int(counts.get("positive", 0))
                neg_cnt = int(counts.get("negative", 0))
                neu_cnt = int(counts.get("neutral", 0))
                tot = pos_cnt + neg_cnt + neu_cnt

                if tot > 0:
                    pos_pct = (pos_cnt / tot) * 100
                    neg_pct = (neg_cnt / tot) * 100
                    neu_pct = (neu_cnt / tot) * 100

                    st.markdown(
                        f"""
                        <div style="background:#111827; border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:16px 20px; margin-top:14px;">
                            <div style="font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:10px;">
                                Total Articles Processed: {tot}
                            </div>
                            <!-- Visual Progress Distribution -->
                            <div style="display:flex; height:10px; border-radius:5px; overflow:hidden; margin-bottom:14px; background:#1E293B;">
                                <div style="width:{pos_pct}%; background:#10B981;" title="Positive {pos_pct:.1f}%"></div>
                                <div style="width:{neu_pct}%; background:#64748B;" title="Neutral {neu_pct:.1f}%"></div>
                                <div style="width:{neg_pct}%; background:#F43F5E;" title="Negative {neg_pct:.1f}%"></div>
                            </div>
                            <div style="display:grid; grid-template-columns:repeat(3, 1fr); text-align:center;">
                                <div>
                                    <div style="color:#34D399; font-weight:700; font-family:'JetBrains Mono'; font-size:1.1rem;">{pos_cnt}</div>
                                    <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase;">Positive ({pos_pct:.0f}%)</div>
                                </div>
                                <div>
                                    <div style="color:#94A3B8; font-weight:700; font-family:'JetBrains Mono'; font-size:1.1rem;">{neu_cnt}</div>
                                    <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase;">Neutral ({neu_pct:.0f}%)</div>
                                </div>
                                <div>
                                    <div style="color:#FB7185; font-weight:700; font-family:'JetBrains Mono'; font-size:1.1rem;">{neg_cnt}</div>
                                    <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase;">Negative ({neg_pct:.0f}%)</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ──────────────────────────────────────────────────────────
    # TAB 4: AI STRATEGIST
    # ──────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC;">Autonomous Market Synthesis</div>
                    <div style="font-size:0.78rem; color:#64748B;">LLM-powered executive reasoning based on price action, indicators &amp; sentiment data.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Prepare context for AI generator
        sentiment_summary = {
            "sentiment_mean": latest_sent_val,
            "n_positive": int(scored_df[scored_df["label"] == "positive"].shape[0]) if not scored_df.empty else 0,
            "n_negative": int(scored_df[scored_df["label"] == "negative"].shape[0]) if not scored_df.empty else 0,
            "n_neutral": int(scored_df[scored_df["label"] == "neutral"].shape[0]) if not scored_df.empty else 0,
        }

        top_headlines_for_llm: list[dict] = []
        if not scored_df.empty:
            temp = scored_df.copy()
            if "date" in temp.columns:
                temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
                temp = temp.sort_values("date", ascending=False)
            top_headlines_for_llm = temp.head(5).to_dict(orient="records")

        best_result = results.get("XGBoost", results.get("Linear Regression", results.get("Naive Baseline")))
        next_price_for_llm: Optional[float] = None
        if best_result is not None:
            try:
                next_price_for_llm = predict_next_day(best_result, feature_df.iloc[-1])
            except Exception:
                pass

        if st.button("✨ Generate Strategic AI Briefing", type="primary"):
            with st.spinner("Synthesizing quantitative indicators and news flow…"):
                summary = generate_summary(
                    ticker=ticker,
                    recent_prices=price_df.tail(5),
                    feature_df=feature_df,
                    sentiment_summary=sentiment_summary,
                    top_headlines=top_headlines_for_llm,
                    next_day_forecast=next_price_for_llm,
                    selected_model=best_result.model_name if best_result else "XGBoost",
                )
                # If API call returned a fallback or error, synthesize a rich quant brief
                if "Unable to generate AI summary" in summary or "API key not configured" in summary:
                    dir_word = "bullish" if (next_price_for_llm and next_price_for_llm >= latest['Close']) else "bearish"
                    ma50_state = "above" if (ma50_val and latest['Close'] > ma50_val) else "below"
                    summary = (
                        f"**{ticker} Market Synthesis:** The stock closed at **${latest['Close']:.2f}** ({chg_sign}{pct_change:.2f}%), "
                        f"trading {ma50_state} its 50-day moving average with an RSI(14) reading of **{rsi_val:.1f}** ({rsi_label.lower()} territory). "
                        f"FinBERT news sentiment registered a net score of **{latest_sent_val:+.3f}** across {len(scored_df)} tracked headlines. "
                        f"The {best_result.model_name if best_result else 'quantitative'} ensemble projects a **{dir_word}** continuation "
                        f"toward **${next_price_for_llm:.2f}** over the next trading session.\n\n"
                        "This is not financial advice."
                    )
                st.session_state["ai_summary"] = summary

        if "ai_summary" in st.session_state:
            st.markdown(
                f"""
                <div class="ai-briefing-card">
                    <div class="ai-briefing-title">
                        <span>✦</span> Executive Intelligence Briefing // {ticker}
                    </div>
                    <div class="ai-briefing-content">
                        {st.session_state["ai_summary"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Click **✨ Generate Strategic AI Briefing** to produce an executive synthesis of technical indicators and sentiment vectors.")

    # ──────────────────────────────────────────────────────────
    # TAB 5: QUANT BENCHMARK
    # ──────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown(
            '<div style="font-size:0.88rem; font-weight:700; color:#CBD5E1; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:12px;">Model Performance &amp; Statistical Validation</div>',
            unsafe_allow_html=True,
        )

        results_list = list(results.values())
        if results_list:
            metrics_df = build_metrics_table(results_list)

            # Crown top model
            if "RMSE" in metrics_df.columns:
                best_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(90deg, rgba(2,132,199,0.15) 0%, rgba(16,185,129,0.1) 100%); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:14px 20px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="font-size:1.4rem;">🏆</span>
                            <div>
                                <div style="font-weight:700; font-size:0.95rem; color:#F8FAFC;">Top-Performing Estimator: {best_row['Model']}</div>
                                <div style="font-size:0.75rem; color:#94A3B8;">Achieved lowest out-of-sample RMSE ({best_row['RMSE']:.4f}) and {best_row['Dir. Accuracy (%)']:.1f}% directional accuracy.</div>
                            </div>
                        </div>
                        <span class="badge badge-cyan">BENCHMARK WINNER</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.plotly_chart(metrics_bar_chart(metrics_df), use_container_width=True)
            with col_m2:
                st.plotly_chart(directional_accuracy_chart(metrics_df), use_container_width=True)

            # Feature Importance for XGBoost or Linear Regression
            xgb_or_linear = results.get("XGBoost", results.get("Linear Regression"))
            if xgb_or_linear is not None:
                fi_fig = feature_importance_chart(xgb_or_linear)
                if fi_fig is not None:
                    st.plotly_chart(fi_fig, use_container_width=True)

            # Walk-forward cross validation
            st.markdown(
                '<div style="font-size:0.82rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-top:16px; margin-bottom:8px;">Walk-Forward Cross-Validation (5 TimeSeries Folds)</div>',
                unsafe_allow_html=True,
            )
            for r in results_list:
                if r.cv_scores:
                    mean_cv = np.mean(r.cv_scores)
                    std_cv = np.std(r.cv_scores)
                    st.markdown(f"• **{r.model_name}**: &nbsp;`RMSE {mean_cv:.4f} ± {std_cv:.4f}`")

            st.markdown(
                """
                <div style="margin-top:20px; padding:14px 18px; background:#111827; border-radius:10px; border:1px solid rgba(255,255,255,0.06); font-size:0.78rem; color:#94A3B8; line-height:1.6;">
                    <strong style="color:#F8FAFC;">Methodology &amp; Zero Look-Ahead Bias Guarantee:</strong><br/>
                    All evaluation metrics are computed strictly on the out-of-sample test partition (the most recent 20% of trading sessions). StandardScalers are fitted exclusively on training slices to ensure no future leakage.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
