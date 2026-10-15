"""
evaluate.py
===========
Model evaluation utilities:
  - RMSE
  - MAPE (handles near-zero actuals)
  - Directional accuracy (did the model call the direction right?)
  - Comparison table across all fitted models vs the naive baseline
  - Plotly bar chart of metrics
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models import FitResult

logger = logging.getLogger(__name__)


# ─────────────────────────── metric functions ─────────────────

def rmse(actuals: np.ndarray, preds: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actuals - preds) ** 2)))


def mape(actuals: np.ndarray, preds: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    Actuals near zero are guarded with eps to avoid division by zero.
    """
    safe_actuals = np.where(np.abs(actuals) < eps, eps, actuals)
    return float(np.mean(np.abs((safe_actuals - preds) / safe_actuals)) * 100)


def directional_accuracy(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Percentage of days where the predicted direction matches actual.
    Direction = sign(today_pred - yesterday_actual).
    Requires at least 2 data points.
    """
    if len(actuals) < 2:
        return float("nan")
    actual_dir = np.sign(np.diff(actuals))
    pred_dir = np.sign(np.diff(preds))
    return float(np.mean(actual_dir == pred_dir) * 100)


# ─────────────────────────── per-model evaluation ─────────────

def evaluate_result(result: FitResult, split: str = "test") -> dict[str, Any]:
    """
    Compute all metrics for a single FitResult.

    Parameters
    ----------
    result : FitResult from models.py
    split  : 'train' or 'test'

    Returns
    -------
    dict with keys: model_name, rmse, mape, dir_acc
    """
    if split == "test":
        actuals = result.test_actuals
        preds = result.test_preds
    else:
        actuals = result.train_actuals
        preds = result.train_preds

    return {
        "Model": result.model_name,
        "RMSE": round(rmse(actuals, preds), 4),
        "MAPE (%)": round(mape(actuals, preds), 4),
        "Dir. Accuracy (%)": round(directional_accuracy(actuals, preds), 2),
    }


def build_metrics_table(results: list[FitResult]) -> pd.DataFrame:
    """
    Build a metrics comparison DataFrame for all models.

    The first model in *results* is treated as the baseline.
    Columns added: RMSE vs Baseline, MAPE vs Baseline.

    Returns
    -------
    pd.DataFrame with one row per model and metric columns.
    """
    rows = [evaluate_result(r) for r in results]
    df = pd.DataFrame(rows)

    if len(df) > 0:
        baseline_rmse = df.iloc[0]["RMSE"]
        baseline_mape = df.iloc[0]["MAPE (%)"]
        df["RMSE vs Baseline"] = (df["RMSE"] - baseline_rmse).round(4)
        df["MAPE vs Baseline (%)"] = (df["MAPE (%)"] - baseline_mape).round(4)

    logger.info("Metrics table:\n%s", df.to_string(index=False))
    return df


# ─────────────────────────── charts ───────────────────────────

def metrics_bar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Side-by-side bar chart of RMSE and MAPE per model with modern styling.
    """
    fig = go.Figure()
    # Refined fintech palette: crisp blue & soft cyan
    colors = {
        "RMSE": "#3B82F6",
        "MAPE (%)": "#06B6D4",
    }

    for metric in ["RMSE", "MAPE (%)"]:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=metrics_df["Model"],
                y=metrics_df[metric],
                marker_color=colors[metric],
                marker_line_width=0,
                text=metrics_df[metric].apply(lambda v: f"{v:.3f}"),
                textposition="outside",
                textfont=dict(size=11, color="#94A3B8"),
            )
        )

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        title=dict(
            text="Model Comparison — RMSE & MAPE",
            font=dict(size=14, color="#F1F5F9", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            title="",
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#CBD5E1", size=12),
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8", size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1", size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def directional_accuracy_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart showing directional accuracy per model.
    """
    colors = [
        "#10B981" if v >= 50 else "#F43F5E"
        for v in metrics_df["Dir. Accuracy (%)"]
    ]

    fig = go.Figure(
        go.Bar(
            x=metrics_df["Dir. Accuracy (%)"],
            y=metrics_df["Model"],
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=metrics_df["Dir. Accuracy (%)"].apply(lambda v: f"{v:.1f}%"),
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=11, family="Inter, system-ui, sans-serif"),
        )
    )
    fig.add_vline(
        x=50,
        line_dash="dot",
        line_color="rgba(148, 163, 184, 0.5)",
        annotation_text="50% baseline",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#94A3B8"),
    )
    fig.update_layout(
        title=dict(
            text="Directional Accuracy (%)",
            font=dict(size=14, color="#F1F5F9", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            title="",
            range=[0, 100],
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94A3B8", size=11),
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#CBD5E1", size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def forecast_vs_actual_chart(
    result: FitResult,
    price_df: pd.DataFrame,
) -> go.Figure:
    """
    Line chart: Actual next-day close vs model predictions on the test set.
    """
    fig = go.Figure()

    # Full close history (subtle background line)
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=price_df["Close"],
            mode="lines",
            name="History",
            line=dict(color="rgba(148, 163, 184, 0.3)", width=1.2),
            showlegend=True,
        )
    )

    # Actual test-set next-day closes
    fig.add_trace(
        go.Scatter(
            x=result.test_index,
            y=result.test_actuals,
            mode="lines",
            name="Actual Close",
            line=dict(color="#10B981", width=2.2),
        )
    )

    # Model predictions
    fig.add_trace(
        go.Scatter(
            x=result.test_index,
            y=result.test_preds,
            mode="lines",
            name=f"{result.model_name} Forecast",
            line=dict(color="#38BDF8", width=2.2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Forecast vs Actual — {result.model_name}",
            font=dict(size=15, color="#F1F5F9", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            title="",
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8", size=11),
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8", size=11),
            tickprefix="$",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1", size=12),
        ),
    )
    return fig


def feature_importance_chart(result: FitResult) -> Optional[go.Figure]:
    """
    Horizontal bar chart showing the top feature importances or model weights.
    Supports tree-based models (e.g. XGBoost) and linear models (Ridge).
    """
    if result is None or not result.feature_cols:
        return None

    importances = None
    metric_label = "Feature Importance"

    if hasattr(result.model, "feature_importances_"):
        importances = result.model.feature_importances_
        metric_label = "Gain Importance"
    elif hasattr(result.model, "coef_"):
        importances = np.abs(result.model.coef_)
        metric_label = "Absolute Ridge Weight (|β|)"

    if importances is None or len(importances) != len(result.feature_cols):
        return None

    series = pd.Series(importances, index=result.feature_cols).sort_values(ascending=True)
    top_series = series.tail(10)

    fig = go.Figure(
        go.Bar(
            x=top_series.values,
            y=top_series.index,
            orientation="h",
            marker=dict(
                color=top_series.values,
                colorscale=[[0, "#1E3A8A"], [1, "#38BDF8"]],
                line=dict(width=0),
            ),
            text=[f"{v:.4f}" for v in top_series.values],
            textposition="auto",
            textfont=dict(color="#F1F5F9", size=11, family="Inter, system-ui, sans-serif"),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Top Predictive Signals ({metric_label}) — {result.model_name}",
            font=dict(size=14, color="#F1F5F9", family="Inter, system-ui, sans-serif"),
        ),
        xaxis=dict(
            title="",
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94A3B8", size=10),
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#CBD5E1", size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig

