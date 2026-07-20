"""Feature engineering — listing-time information only (leakage guard).

The one rule: every feature must be knowable at the moment a broker looks at
the listing. Nothing derived from the realized/future price enters X.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "avg_price"
NUMERIC = ["days_to_event", "weekend", "event_score", "performer_score",
           "log_listings", "near_event", "score_x_near"]
# present only when the source can provide them (e.g. venue_tier from the
# Ticketmaster connector's leave-one-out venue price level)
OPTIONAL_NUMERIC = ["venue_tier"]
CATEGORICAL = ["taxonomy"]
META = ["event_id", "title", "venue_city"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns for this frame (optional features if present)."""
    return (CATEGORICAL + NUMERIC
            + [c for c in OPTIONAL_NUMERIC if c in df.columns])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the contract and derive model features."""
    required = {"event_id", "taxonomy", "days_to_event", "weekend",
                "event_score", "performer_score", "listing_count", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input missing columns: {sorted(missing)}")

    out = df.copy()
    out = out[(out[TARGET] > 0) & (out["days_to_event"] >= 0)].reset_index(drop=True)
    out["log_listings"] = np.log1p(out["listing_count"])
    out["near_event"] = (out["days_to_event"] <= 14).astype(int)
    # hot events behave differently close to the date — give the model the seam
    out["score_x_near"] = out["event_score"] * out["near_event"]
    return out
