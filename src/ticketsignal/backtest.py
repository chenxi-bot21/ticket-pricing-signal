"""Walk-forward backtest of the deal signal on a longitudinal panel.

True validation of a pricing signal is temporal: fire the signal at time t
using only information available at t, then watch what prices actually do
next. Single-snapshot cross-validation cannot answer that; a panel can.

Synthetic panel: weekly snapshots of each event's listing price as the event
approaches. Fair value moves with the same drivers as the snapshot generator;
the observed price is fair value times a persistent-but-mean-reverting
mispricing term (AR(1) in log space). A working signal buys below-fair
listings and earns the reversion; a broken one shows a flat decile ramp.

Protocol (leakage-safe by construction):
- expanding window — at week t the fair-value model is fit on snapshots
  strictly before t;
- score week-t listings, bucket deal scores into deciles;
- forward return = price(t+1) / price(t) - 1;
- report the per-decile mean forward return, weekly rank IC (Spearman), and
  the top-minus-bottom spread. Monotone ramp + positive IC = signal works.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from .data import GENRES, _BASE
from .features import build_features, CATEGORICAL, NUMERIC
from .model import _pipeline


# --------------------------------------------------------------------------
# Longitudinal synthetic market
# --------------------------------------------------------------------------
def simulate_panel(n_events: int = 600, n_weeks: int = 26,
                   seed: int = 42, rho: float = 0.7,
                   mispricing_vol: float = 0.10) -> pd.DataFrame:
    """Weekly panel of event listings with AR(1) mispricing around fair value."""
    rng = np.random.default_rng(seed)
    genre = rng.choice(GENRES, n_events, p=[.30, .12, .10, .12, .08, .13, .08, .07])
    popularity = np.clip(rng.beta(2.2, 3.5, n_events), 0.02, 0.99)
    performer = np.clip(popularity + rng.normal(0, .08, n_events), 0.01, 1.0)
    weekend = rng.integers(0, 2, n_events)
    start_days = rng.uniform(60, 60 + 7 * n_weeks, n_events)   # staggered dates
    listings0 = np.maximum(
        5, rng.poisson(40 + 400 * rng.beta(1.6, 4, n_events))).astype(int)
    base = np.array([_BASE[g] for g in genre], dtype=float)
    city = rng.choice(["New York", "Los Angeles", "Chicago", "Austin",
                       "Boston", "Seattle", "Miami", "Denver"], n_events)

    eps = rng.normal(0, mispricing_vol, n_events)   # initial mispricing
    rows = []
    for week in range(n_weeks):
        days = start_days - 7 * week
        alive = days >= 1
        # supply drains ~35% into the event, with noise
        frac_gone = np.clip(1 - days / start_days, 0, 1)
        listings = np.maximum(
            3, (listings0 * (1 - 0.35 * frac_gone)
                * rng.lognormal(0, 0.05, n_events))).astype(int)
        pop_mult = 0.45 + 1.8 * popularity ** 1.4
        supply_mult = (listings / 150.0) ** -0.18
        time_mult = 1.0 + (popularity - 0.45) * (1.0 - days / 120.0) * 0.8
        wk_mult = 1.0 + 0.08 * weekend
        fair = base * pop_mult * supply_mult * time_mult * wk_mult
        # AR(1) mispricing: persistent enough to catch, mean-reverting enough to pay
        eps = rho * eps + rng.normal(0, mispricing_vol * np.sqrt(1 - rho ** 2),
                                     n_events)
        price = fair * np.exp(eps)
        for i in np.where(alive)[0]:
            rows.append({
                "week": week,
                "event_id": i + 1,
                "title": f"{genre[i].title()} event #{i + 1}",
                "taxonomy": genre[i],
                "days_to_event": round(float(days[i]), 2),
                "weekend": int(weekend[i]),
                "event_score": round(float(popularity[i]), 4),
                "performer_score": round(float(performer[i]), 4),
                "listing_count": int(listings[i]),
                "venue_city": city[i],
                "avg_price": round(float(price[i]), 2),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Walk-forward protocol
# --------------------------------------------------------------------------
@dataclass
class BacktestReport:
    metrics: dict
    deciles: pd.DataFrame
    weekly: pd.DataFrame = field(repr=False)


def walk_forward_backtest(panel: pd.DataFrame, min_train_weeks: int = 8,
                          n_buckets: int = 10) -> BacktestReport:
    """Expanding-window refit; score week t; realize returns over week t+1."""
    nxt = panel[["week", "event_id", "avg_price"]].copy()
    nxt["week"] -= 1
    nxt = nxt.rename(columns={"avg_price": "next_price"})

    picks = []
    ics = []
    for t in range(min_train_weeks, int(panel["week"].max())):
        train = build_features(panel[panel["week"] < t])
        test = build_features(panel[panel["week"] == t]).merge(
            nxt[nxt["week"] == t], on=["week", "event_id"], how="inner")
        if len(test) < 30:
            continue
        pipe = _pipeline().fit(train[CATEGORICAL + NUMERIC],
                               np.log(train["avg_price"].to_numpy()))
        fair = np.exp(pipe.predict(test[CATEGORICAL + NUMERIC]))
        test = test.assign(
            deal_score=100 * (fair - test["avg_price"]) / fair,
            fwd_ret=test["next_price"] / test["avg_price"] - 1.0)
        ics.append(float(sps.spearmanr(test["deal_score"],
                                       test["fwd_ret"]).statistic))
        picks.append(test[["week", "event_id", "taxonomy", "deal_score",
                           "fwd_ret"]])

    weekly = pd.concat(picks, ignore_index=True)
    weekly["bucket"] = pd.qcut(weekly["deal_score"], n_buckets,
                               labels=False, duplicates="drop") + 1
    deciles = (weekly.groupby("bucket")
               .agg(mean_deal_score=("deal_score", "mean"),
                    mean_fwd_ret_pct=("fwd_ret", lambda s: 100 * s.mean()),
                    n=("fwd_ret", "size"))
               .round(3).reset_index())

    ic = np.array(ics)
    top = float(deciles.loc[deciles["bucket"].idxmax(), "mean_fwd_ret_pct"])
    bot = float(deciles.loc[deciles["bucket"].idxmin(), "mean_fwd_ret_pct"])
    metrics = {
        "weeks_tested": int(len(ic)),
        "n_signal_obs": int(len(weekly)),
        "rank_ic_mean": round(float(ic.mean()), 4),
        "rank_ic_tstat": round(float(ic.mean() / (ic.std(ddof=1)
                               / np.sqrt(len(ic)))), 2),
        "top_decile_fwd_ret_pct": round(top, 3),
        "bottom_decile_fwd_ret_pct": round(bot, 3),
        "top_minus_bottom_pct": round(top - bot, 3),
    }
    return BacktestReport(metrics=metrics, deciles=deciles, weekly=weekly)
