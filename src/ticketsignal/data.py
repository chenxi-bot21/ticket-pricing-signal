"""Data layer: SeatGeek public API client + seeded synthetic market.

Every row = one event snapshot at "listing time". Columns produced by both
sources (the model contract):

    event_id, title, taxonomy, days_to_event, weekend,
    event_score, performer_score, listing_count, venue_city,
    avg_price  (target, USD)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

SEATGEEK_URL = "https://api.seatgeek.com/2/events"

GENRES = ["concert", "nba", "nfl", "mlb", "nhl", "theater", "comedy", "family"]
# Synthetic base prices per genre (rough real-world flavour).
_BASE = {"concert": 120, "nba": 140, "nfl": 180, "mlb": 60, "nhl": 90,
         "theater": 100, "comedy": 70, "family": 55}


# --------------------------------------------------------------------------
# SeatGeek (live) — free client id from seatgeek.com/build
# --------------------------------------------------------------------------
def fetch_seatgeek_events(client_id: str, pages: int = 5,
                          per_page: int = 100) -> pd.DataFrame:
    """Pull upcoming events with price stats from the public SeatGeek API.

    Filters at the API level: future events only, with at least one listing,
    sorted by popularity (popular events are the ones with market stats).

    Field note (2026-07, verified empirically): SeatGeek gates the price
    stats — ``average_price``/``listing_count`` return null for new client
    ids (partner-level apps only). Events without a price are skipped, so a
    non-partner id yields an empty frame. The connector is kept because the
    pipeline is source-agnostic: any priced feed matching the same column
    contract (see module docstring) drops in unchanged.
    """
    rows = []
    now = datetime.now(timezone.utc)
    for page in range(1, pages + 1):
        resp = requests.get(SEATGEEK_URL, params={
            "client_id": client_id, "per_page": per_page, "page": page,
            "datetime_utc.gte": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "listing_count.gt": 0,
            "sort": "score.desc",
        }, timeout=30)
        resp.raise_for_status()
        for ev in resp.json().get("events", []):
            stats = ev.get("stats") or {}
            avg = stats.get("average_price")
            if not avg or avg <= 0:
                continue                      # no market → nothing to model
            when = ev.get("datetime_utc")
            try:
                dt = datetime.fromisoformat(when).replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            days = (dt - now).total_seconds() / 86400.0
            if days < 0:
                continue
            taxonomy = "other"
            if ev.get("taxonomies"):
                taxonomy = ev["taxonomies"][0].get("name", "other")
            performers = ev.get("performers") or []
            rows.append({
                "event_id": ev.get("id"),
                "title": ev.get("title", ""),
                "taxonomy": taxonomy,
                "days_to_event": round(days, 2),
                "weekend": int(dt.weekday() >= 4),
                "event_score": float(ev.get("score") or 0.0),
                "performer_score": max((float(p.get("score") or 0.0)
                                        for p in performers), default=0.0),
                "listing_count": int(stats.get("listing_count") or 0),
                "venue_city": (ev.get("venue") or {}).get("city", ""),
                "avg_price": float(avg),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Synthetic market (seeded) — runs offline, drivers known by construction
# --------------------------------------------------------------------------
def synthetic_events(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Seeded synthetic snapshot with realistic price drivers.

    Known structure (so estimator recovery is testable): price rises with
    popularity, falls with supply, decays as low-demand events approach the
    date while hot events climb near the date; weekend premium ~8%.
    """
    rng = np.random.default_rng(seed)
    genre = rng.choice(GENRES, n, p=[.30, .12, .10, .12, .08, .13, .08, .07])
    popularity = np.clip(rng.beta(2.2, 3.5, n), 0.02, 0.99)     # event_score
    performer = np.clip(popularity + rng.normal(0, .08, n), 0.01, 1.0)
    days = rng.uniform(1, 120, n)
    weekend = rng.integers(0, 2, n)
    listings = np.maximum(
        5, rng.poisson(40 + 400 * rng.beta(1.6, 4, n))).astype(int)

    base = np.array([_BASE[g] for g in genre], dtype=float)
    pop_mult = 0.45 + 1.8 * popularity ** 1.4
    supply_mult = (listings / 150.0) ** -0.18
    # time-to-event: cold events decay into the date, hot events appreciate
    time_mult = 1.0 + (popularity - 0.45) * (1.0 - days / 120.0) * 0.8
    wk_mult = 1.0 + 0.08 * weekend
    noise = rng.lognormal(0, 0.16, n)

    price = base * pop_mult * supply_mult * time_mult * wk_mult * noise
    return pd.DataFrame({
        "event_id": np.arange(1, n + 1),
        "title": [f"{g.title()} event #{i}" for i, g in enumerate(genre, 1)],
        "taxonomy": genre,
        "days_to_event": np.round(days, 2),
        "weekend": weekend,
        "event_score": np.round(popularity, 4),
        "performer_score": np.round(performer, 4),
        "listing_count": listings,
        "venue_city": rng.choice(
            ["New York", "Los Angeles", "Chicago", "Austin", "Boston",
             "Seattle", "Miami", "Denver"], n),
        "avg_price": np.round(price, 2),
    })
