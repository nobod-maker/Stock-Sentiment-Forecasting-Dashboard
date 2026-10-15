"""
models.py
=========
Trains and stores four forecasting models:
  (a) NaiveBaseline      — tomorrow = today's close
  (b) LinearRegression   — sklearn Ridge regression
  (c) XGBoostRegressor   — gradient boosting
  (d) LSTMForecaster     — PyTorch LSTM (optional stretch goal)

Design principles
-----------------
* Scalers are fitted ONLY on training data (no leakage).
* Targets shifted by -1 day (done in features.py via Next_Close).
* TimeSeriesSplit cross-validation available for hyper-parameter insight.
* All models expose a uniform .predict(X) API.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────── data types ───────────────────────

@dataclass
class FitResult:
    """Container returned by every model's fit() function."""
    model_name: str
    model: object                          # fitted estimator
    scaler: Optional[StandardScaler]       # fitted scaler (or None)
    feature_cols: list[str]
    train_preds: np.ndarray
    test_preds: np.ndarray
    train_actuals: np.ndarray
    test_actuals: np.ndarray
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    cv_scores: list[float] = field(default_factory=list)


# ─────────────────────────── (a) Naive baseline ───────────────

class NaiveBaseline:
    """Predict tomorrow's close = today's close (random walk baseline)."""

    name = "Naive Baseline"

    def predict(self, X: np.ndarray, close_values: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return close_values unchanged as the forecast."""
        return close_values.copy()


def fit_naive(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Next_Close",
) -> FitResult:
    model = NaiveBaseline()
    train_preds = train_df["Close"].values
    test_preds = test_df["Close"].values
    return FitResult(
        model_name=model.name,
        model=model,
        scaler=None,
        feature_cols=["Close"],
        train_preds=train_preds,
        test_preds=test_preds,
        train_actuals=train_df[target_col].values,
        test_actuals=test_df[target_col].values,
        train_index=train_df.index,
        test_index=test_df.index,
    )


# ─────────────────────────── (b) Linear Regression ───────────

def fit_linear(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Next_Close",
    alpha: float = 1.0,
) -> FitResult:
    """
    Fit a Ridge regression model (L2 regularisation).

    Scaler is fitted on TRAIN only, then applied to TEST.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # CV scores for transparency (fit on train only)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores: list[float] = []
    for tr_idx, val_idx in tscv.split(X_train):
        m = Ridge(alpha=alpha)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        preds = m.predict(X_train[val_idx])
        rmse = np.sqrt(np.mean((y_train[val_idx] - preds) ** 2))
        cv_scores.append(float(rmse))

    logger.info("Linear Regression CV RMSE: %.4f ± %.4f", np.mean(cv_scores), np.std(cv_scores))

    return FitResult(
        model_name="Linear Regression",
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        train_preds=model.predict(X_train),
        test_preds=model.predict(X_test),
        train_actuals=y_train,
        test_actuals=y_test,
        train_index=train_df.index,
        test_index=test_df.index,
        cv_scores=cv_scores,
    )


# ─────────────────────────── (c) XGBoost ─────────────────────

def fit_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Next_Close",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> FitResult:
    """
    Fit an XGBoost regressor with sensible default hyper-parameters.

    No scaling needed for tree models; scaler=None in result.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    # Walk-forward validation on training split
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores: list[float] = []
    for tr_idx, val_idx in tscv.split(X_train):
        m = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
        )
        m.fit(X_train[tr_idx], y_train[tr_idx])
        preds = m.predict(X_train[val_idx])
        rmse = np.sqrt(np.mean((y_train[val_idx] - preds) ** 2))
        cv_scores.append(float(rmse))

    # Final fit on full training data
    model.fit(X_train, y_train)
    logger.info("XGBoost CV RMSE: %.4f ± %.4f", np.mean(cv_scores), np.std(cv_scores))

    return FitResult(
        model_name="XGBoost",
        model=model,
        scaler=None,
        feature_cols=feature_cols,
        train_preds=model.predict(X_train),
        test_preds=model.predict(X_test),
        train_actuals=y_train,
        test_actuals=y_test,
        train_index=train_df.index,
        test_index=test_df.index,
        cv_scores=cv_scores,
    )


# ─────────────────────────── (d) LSTM (optional) ─────────────

def fit_lstm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Next_Close",
    seq_len: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> FitResult:
    """
    Train a two-layer LSTM regressor (PyTorch).

    Falls back gracefully if PyTorch is not installed.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("PyTorch is not installed. Run: pip install torch")

    scaler = StandardScaler()
    X_all = np.concatenate([train_df[feature_cols].values, test_df[feature_cols].values], axis=0)
    split = len(train_df)

    X_train_raw = X_all[:split]
    X_test_raw = X_all[split:]
    scaler.fit(X_train_raw)
    X_train_sc = scaler.transform(X_train_raw)
    X_test_sc = scaler.transform(X_test_raw)

    y_train = train_df[target_col].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    # ── sequence builder ──
    def make_sequences(X: np.ndarray, y: np.ndarray, seq: int):
        Xs, ys = [], []
        for i in range(seq, len(X)):
            Xs.append(X[i - seq : i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # Build sequences across the full scaled array so the test window
    # has proper history (no leakage — we only fit scaler on train)
    X_all_sc = np.concatenate([X_train_sc, X_test_sc], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    Xseq, yseq = make_sequences(X_all_sc, y_all, seq_len)

    train_end = split - seq_len
    Xseq_train, yseq_train = Xseq[:train_end], yseq[:train_end]
    Xseq_test, yseq_test = Xseq[train_end:], yseq[train_end:]

    if len(Xseq_train) == 0:
        raise ValueError("Not enough training data for LSTM with seq_len=%d" % seq_len)

    # ── model definition ──
    class _LSTMModel(nn.Module):
        def __init__(self, n_features: int, hidden: int, layers: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train_sc.shape[1]
    net = _LSTMModel(n_features, hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(Xseq_train),
        torch.tensor(yseq_train),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info("LSTM epoch %d/%d — loss %.6f", epoch + 1, epochs, epoch_loss / len(loader))

    net.eval()
    with torch.no_grad():
        train_preds = net(torch.tensor(Xseq_train).to(device)).cpu().numpy()
        test_preds = net(torch.tensor(Xseq_test).to(device)).cpu().numpy()

    return FitResult(
        model_name="LSTM",
        model=net,
        scaler=scaler,
        feature_cols=feature_cols,
        train_preds=train_preds,
        test_preds=test_preds,
        train_actuals=yseq_train,
        test_actuals=yseq_test,
        train_index=train_df.index[seq_len:],
        test_index=test_df.index[: len(yseq_test)],
    )


# ─────────────────────────── next-day forecast ────────────────

def predict_next_day(
    result: FitResult,
    latest_row: pd.Series,
) -> float:
    """
    Generate a next-day price forecast from the latest feature row.

    Parameters
    ----------
    result     : Fitted FitResult from any model.
    latest_row : The most recent row of the feature DataFrame.

    Returns
    -------
    Predicted next-day closing price as a float.
    """
    model = result.model
    if isinstance(model, NaiveBaseline):
        return float(latest_row["Close"])

    X = latest_row[result.feature_cols].values.reshape(1, -1)

    if result.scaler is not None:
        X = result.scaler.transform(X)

    # LSTM requires sequence input — skip for now in next-day forecast
    if hasattr(model, "parameters"):  # PyTorch nn.Module
        logger.warning("LSTM next-day forecast requires sequence context; returning last test pred.")
        return float(result.test_preds[-1]) if len(result.test_preds) > 0 else float(latest_row["Close"])

    return float(model.predict(X)[0])
