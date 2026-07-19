# Ticket Fair-Value Signal

A proof-of-concept **pricing signal for the secondary ticket market**: estimate a
fair market price for live-event tickets from listing-time information, then
flag listings trading **below fair value (potential buys)** or **above it
(rich)** — the model that sits behind a broker's buy/sell decision.

Built as a compact demonstration of the modeling discipline that matters in
production forecasting:

- **Leakage control** — features are restricted to what is knowable at listing
  time (time to event, popularity, supply, genre, venue). Nothing derived from
  the realized price enters the model.
- **Honest evaluation** — every event's fair value is an **out-of-sample**
  prediction (5-fold CV), reported against two naive baselines (global mean,
  per-genre mean). The tests fail if the model can't beat the baselines.
- **Explainability** — permutation importance on a held-out fold; in the
  synthetic market the true drivers are known by construction, so estimator
  recovery is a testable property, not a hope.

## Data

| Source | What it is |
|---|---|
| **Ticketmaster Discovery API** (connector) | Real upcoming US events with real `priceRanges` — the only freely accessible price signal found in due diligence (free key at developer.ticketmaster.com). Also parses two demand proxies from the same payload: artist tour size and venue activity. Terms don't allow redistributing pulled data, so no dataset ships here — run it with your own key. |
| **Synthetic market** (offline, default) | Seeded generator with realistic drivers: popularity premium, supply discount, time-to-event dynamics (cold events decay, hot events climb), weekend effect. Because the true drivers are known by construction, estimator recovery is testable. |
| **SeatGeek public API** (connector) | Upcoming events with popularity scores and timing. **Field note from due diligence:** SeatGeek now gates the price stats (`average_price`, `listing_count`) to partner-level apps — verified empirically; new client ids receive event metadata but null prices. The connector is written and the pipeline is source-agnostic, so any priced feed (a marketplace partner API, or a broker's own data) drops in via the same contract. |
| **Cached CSV** | Any raw pull (or any dataset matching the column contract) re-runs offline via `--source csv` / the dashboard uploader. |

## Real-data experiment (Ticketmaster, 5,498 events)

Pulled 5,498 unique priced US events (all 50 states, 12 months out, 15+
genres) and ran the identical pipeline twice:

| Round | Features | MAE | Lift vs genre-mean | R² |
|---|---|---|---|---|
| 1 | public only: genre, city, timing | $24.2 | 15% | 0.04 |
| 2 | + two crude demand proxies (artist tour size, venue activity) | **$18.1** | **37%** | **0.30** |

Two findings. **The demand signal is where the power is** — even crude
public proxies triple the model's edge, and permutation importance
concentrates on them (0.48 + 0.30); a marketplace's own demand data (sales
velocity, transactions) is the upgrade path. And the uncertainty band stays
honest on real data: **78.5% measured vs 80% nominal** coverage.

## Quickstart

```bash
pip install -e ".[app]"

# offline demo (no key needed)
python -m ticketsignal.cli --source synthetic

# live data
python -m ticketsignal.cli --source seatgeek --client-id YOUR_ID

# dashboard
streamlit run app.py

# tests
python -m unittest discover -s tests -t .
```

The CLI writes `artifacts/events_scored.csv` (per-event fair price + deal
score), `importance.csv`, and `metrics.json`.

## Deal score

`deal_score_pct = (fair − observed) / fair × 100` — positive means the market
lists the event below the model's fair value. Filter by `listing_count` for
liquidity before acting on it.

## Walk-forward backtest (`--backtest` / dashboard tab)

Cross-sectional accuracy is necessary but not sufficient — a deal score is
only real if prices subsequently move toward fair value. The backtest runs
the temporal protocol on a longitudinal panel (weekly snapshots, AR(1)
mean-reverting mispricing): **expanding-window refit** (each week's model
sees only the past), score week *t*, realize returns over week *t+1*, then
report the decile ramp and weekly rank IC.

Synthetic-panel result: rank IC ≈ 0.32 (t ≈ 20), monotone decile ramp,
top-minus-bottom ≈ 9%/week — the signal harvests the planted mean reversion,
which is exactly what the harness is meant to detect. Plug in real listing
history (a broker's own panel) and the same protocol answers the question
that matters: *does buying the flagged listings make money?*

## Honest limitations

- The P10–P90 band **under-covers** on the synthetic market (~65% empirical vs
  80% nominal) — quantile GBMs are known to be optimistic. The fix is
  conformalized quantile regression (calibrate the band on a held-out fold);
  the dashboard reports measured coverage precisely so this gap is visible
  instead of assumed away.

- One snapshot per event: with longitudinal listing data the same discipline
  extends to time-aware walk-forward validation (fit on past, score the future).
- `average_price` blends seat tiers; production would model section/row level.
- A deal score is a screening signal, not an execution strategy — fees,
  liquidity, and adverse selection all bite in practice.

*Python: pandas · scikit-learn · Streamlit/Plotly. Author: Chenxi Zhao.*
