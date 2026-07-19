"""Ticketmaster Discovery API connector (real events, real price ranges).

Free API key from developer.ticketmaster.com (Discovery API, 5000 req/day).
Unlike SeatGeek (price stats gated to partners), Ticketmaster exposes a
``priceRanges`` min/max on primary inventory for many US events — the only
freely accessible real price signal we found in due diligence.

Two demand proxies are parsed from the same payload and turned into the
contract's score columns as 0-1 percentile ranks:

- ``performer_score``  <- attraction ``upcomingEvents`` total (tour size)
- ``event_score``      <- venue ``upcomingEvents`` total (venue activity)

Measured effect on 5,498 real events (2026-07): adding these two crude
proxies moved the model from a 15% to a 37% MAE lift over the per-genre
baseline (R^2 0.04 -> 0.30) — the demand signal is where the power is.
See README "Real-data experiment".

Note: Ticketmaster's terms do not allow redistributing pulled data, so no
dataset ships with this repo — run the connector with your own key.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

TICKETMASTER_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
SEGMENTS = ["Music", "Sports", "Arts & Theatre", "Family"]


def _parse_event(ev: dict, now: datetime) -> dict | None:
    """Map one Discovery API event to the model contract (or None to skip)."""
    prs = [p for p in (ev.get("priceRanges") or [])
           if p.get("type") == "standard" and p.get("min")]
    if not prs:
        return None
    lo, hi = float(prs[0]["min"]), float(prs[0].get("max") or prs[0]["min"])
    if lo <= 0 or hi < lo:
        return None
    when = (ev.get("dates", {}).get("start", {}) or {}).get("dateTime")
    if not when:
        return None
    try:
        dt = datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError:
        return None
    days = (dt - now).total_seconds() / 86400.0
    if days < 0:
        return None
    cls = (ev.get("classifications") or [{}])[0]
    genre = (cls.get("genre") or {}).get("name") or ""
    taxonomy = (genre if genre and genre != "Undefined"
                else (cls.get("segment") or {}).get("name", "other"))
    emb = ev.get("_embedded") or {}
    venues = emb.get("venues") or [{}]
    attractions = emb.get("attractions") or [{}]
    return {
        "event_id": ev.get("id"),
        "title": ev.get("name", ""),
        "taxonomy": taxonomy.lower(),
        "days_to_event": round(days, 2),
        "weekend": int(dt.weekday() >= 4),
        "attr_upcoming": max(((a.get("upcomingEvents") or {}).get("_total")
                              or 0) for a in attractions),
        "venue_upcoming": ((venues[0].get("upcomingEvents") or {})
                           .get("_total")) or 0,
        "listing_count": 100,             # not exposed by Ticketmaster
        "venue_city": ((venues[0].get("city") or {}).get("name")) or "",
        "avg_price": round((lo + hi) / 2, 2),
    }


def _finalize(rows: list[dict], top_genres: int = 12,
              top_cities: int = 20) -> pd.DataFrame:
    """Bucket long-tail categories; percentile-rank the demand proxies."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = df["taxonomy"].value_counts().head(top_genres).index
    df["taxonomy"] = df["taxonomy"].where(df["taxonomy"].isin(keep), "other")
    keep = df["venue_city"].value_counts().head(top_cities).index
    df["venue_city"] = df["venue_city"].where(df["venue_city"].isin(keep),
                                              "Other")
    df["performer_score"] = df["attr_upcoming"].rank(pct=True).round(4)
    df["event_score"] = df["venue_upcoming"].rank(pct=True).round(4)
    return df.drop(columns=["attr_upcoming", "venue_upcoming"])


def fetch_ticketmaster_events(api_key: str, months: int = 12,
                              pages: int = 5, sleep: float = 0.15,
                              country: str = "US") -> pd.DataFrame:
    """Segment x monthly-window sweep of upcoming priced events."""
    rows, seen = [], set()
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    for seg in SEGMENTS:
        for d in range(0, months * 30, 30):
            for page in range(pages):     # deep-paging cap: (page+1)*200<=1000
                resp = requests.get(TICKETMASTER_URL, params={
                    "apikey": api_key, "size": 200, "page": page,
                    "countryCode": country, "classificationName": seg,
                    "sort": "date,asc",
                    "startDateTime": (now + timedelta(days=d)).strftime(fmt),
                    "endDateTime": (now + timedelta(days=d + 30)).strftime(fmt),
                }, timeout=30)
                resp.raise_for_status()
                events = (resp.json().get("_embedded") or {}).get("events", [])
                for ev in events:
                    row = _parse_event(ev, now)
                    if row and row["event_id"] not in seen:
                        seen.add(row["event_id"])
                        rows.append(row)
                time.sleep(sleep)
                if len(events) < 200:
                    break
    return _finalize(rows)
