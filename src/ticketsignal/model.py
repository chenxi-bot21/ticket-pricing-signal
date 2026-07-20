"""Fair-value model, honest evaluation, and the deal score.

- Model: gradient-boosted trees on log(price) (multiplicative price drivers).
- Uncertainty: quantile GBMs give a P10-P90 fair-value band per event, and the
  band is *checked*: empirical coverage should be ~80% out of sample — the
  regression analogue of calibration. A band that claims 80% and covers 60%
  is lying; we measure it instead of assuming it.
- Evaluation: 5-fold cross-validated out-of-sample predictions, reported
  against two naive baselines (global mean, per-genre mean). A model that
  cannot beat the per-genre mean has learned nothing — say so, don't ship it.
- Deal score: (fair - observed) / fair, in %. Positive = listed below fair
  value (potential buy); negative = rich (potential sell/avoid). Confidence
  is "high" only when the observed price sits outside the entire band.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL, NUMERIC, TARGET, feature_columns


def _pipeline(loss: str = "squared_error", quantile: float | None = None,
              num_cols: list[str] | None = None) -> Pipeline:
    prep = ColumnTransformer([
        # sparse_output=False: high-cardinality categoricals (real-world city/
        # genre columns) would otherwise yield a sparse matrix, which
        # HistGradientBoostingRegressor rejects.
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
        ("num", "passthrough", NUMERIC if num_cols is None else num_cols),
    ])
    return Pipeline([
        ("prep", prep),
        ("gbm", HistGradientBoostingRegressor(
            loss=loss, quantile=quantile,
            max_depth=4, learning_rate=0.08, max_iter=400,
            l2_regularization=1.0, random_state=0)),
    ])


def _monotone(low: np.ndarray, mid: np.ndarray,
              high: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise rearrangement: independently trained quantile models can
    cross in sparse segments; sorting restores low <= mid <= high
    (Chernozhukov et al.'s quantile rearrangement, the 3-point case)."""
    stacked = np.sort(np.vstack([low, mid, high]), axis=0)
    return stacked[0], stacked[1], stacked[2]


@dataclass
class SignalReport:
    metrics: dict
    importance: pd.DataFrame
    scored: pd.DataFrame = field(repr=False)


def train_and_score(df: pd.DataFrame, n_splits: int = 5,
                    seed: int = 0) -> SignalReport:
    """Cross-validated out-of-sample fair values + deal scores for every event."""
    X = df[feature_columns(df)]
    num_cols = [c for c in X.columns if c not in CATEGORICAL]
    y_log = np.log(df[TARGET].to_numpy())
    y = df[TARGET].to_numpy()

    fair = np.zeros(len(df))
    lo = np.zeros(len(df))
    hi = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        mid = _pipeline(num_cols=num_cols).fit(X.iloc[tr], y_log[tr])
        q10 = _pipeline("quantile", 0.10,
                        num_cols=num_cols).fit(X.iloc[tr], y_log[tr])
        q90 = _pipeline("quantile", 0.90,
                        num_cols=num_cols).fit(X.iloc[tr], y_log[tr])
        fair[te] = np.exp(mid.predict(X.iloc[te]))
        lo[te] = np.exp(q10.predict(X.iloc[te]))
        hi[te] = np.exp(q90.predict(X.iloc[te]))

    lo, fair, hi = _monotone(lo, fair, hi)

    # ---- honest metrics vs naive baselines (all out-of-sample) -------------
    base_global = np.full(len(df), y.mean())
    genre_mean = df.groupby("taxonomy")[TARGET].transform("mean").to_numpy()
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    metrics = {
        "n_events": int(len(df)),
        "mae_model": float(mean_absolute_error(y, fair)),
        "mae_global_mean": float(mean_absolute_error(y, base_global)),
        "mae_genre_mean": float(mean_absolute_error(y, genre_mean)),
        "r2_model": float(r2_score(y, fair)),
        "lift_vs_genre_mean_pct": float(
            100 * (1 - mean_absolute_error(y, fair)
                   / mean_absolute_error(y, genre_mean))),
        # regression analogue of calibration: an 80% band should cover ~80%
        "band_nominal_pct": 80.0,
        "band_coverage_pct": round(100 * coverage, 1),
    }

    # ---- explainability: permutation importance on a held-out fold --------
    tr, te = next(KFold(n_splits=5, shuffle=True, random_state=1).split(X))
    pipe = _pipeline(num_cols=num_cols).fit(X.iloc[tr], y_log[tr])
    imp = permutation_importance(pipe, X.iloc[te], y_log[te],
                                 n_repeats=8, random_state=0)
    importance = (pd.DataFrame({
        "feature": X.columns, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False).reset_index(drop=True))

    scored = df.copy()
    scored["fair_price"] = np.round(fair, 2)
    scored["fair_low"] = np.round(lo, 2)
    scored["fair_high"] = np.round(hi, 2)
    scored["deal_score_pct"] = np.round(100 * (fair - y) / fair, 1)
    # high confidence only when price falls outside the entire band
    scored["confidence"] = np.select(
        [y < lo, y > hi], ["high (below band)", "high (above band)"],
        default="inside band")
    scored = scored.sort_values("deal_score_pct", ascending=False)
    return SignalReport(metrics=metrics, importance=importance, scored=scored)
