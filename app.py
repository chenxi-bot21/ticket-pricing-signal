"""
Ticket Fair-Value Signal — interactive dashboard.

    streamlit run app.py

Needs the `app` extras: pip install -e ".[app]"   (streamlit + plotly)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ticketsignal import (build_features, fetch_seatgeek_events,
                          synthetic_events, train_and_score)

st.set_page_config(page_title="Ticket Fair-Value Signal", layout="wide")
st.title("Secondary-Market Ticket Pricing Signal")
st.caption("Proof of concept — leakage-safe fair-value model + deal scores. "
           "Features use listing-time information only; metrics are "
           "out-of-sample (5-fold CV) and compared against naive baselines.")

# ---------------- Sidebar: data + parameters ----------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Synthetic market (offline)",
                                 "SeatGeek API (live)",
                                 "Cached pull (CSV)"])
    if source.startswith("SeatGeek"):
        client_id = st.text_input("SeatGeek client id", type="password",
                                  help="Free at seatgeek.com/build")
        pages = st.slider("Pages (100 events/page)", 1, 10, 5)
        if not client_id:
            st.info("Enter a client id to pull live events.")
            st.stop()
    elif source.startswith("Cached"):
        upload = st.file_uploader("Raw pull CSV (from cli --save-raw)", type="csv")
        if upload is None:
            st.info("Upload a cached raw pull to score it offline.")
            st.stop()
    else:
        n = st.slider("Synthetic events", 300, 5000, 1500, step=100)
        seed = st.number_input("Seed", 1, 9999, 42)

    st.header("Deal filter")
    min_listings = st.slider("Min listings (liquidity)", 0, 200, 20)


@st.cache_data(show_spinner="Scoring events…")
def _run(source_key: str, **kw):
    if source_key == "live":
        raw = fetch_seatgeek_events(kw["client_id"], pages=kw["pages"])
    elif source_key == "csv":
        raw = kw["frame"]
    else:
        raw = synthetic_events(n=kw["n"], seed=int(kw["seed"]))
    feats = build_features(raw)
    rep = train_and_score(feats)
    return rep.metrics, rep.importance, rep.scored


if source.startswith("SeatGeek"):
    metrics, importance, scored = _run("live", client_id=client_id, pages=pages)
elif source.startswith("Cached"):
    metrics, importance, scored = _run("csv", frame=pd.read_csv(upload))
else:
    metrics, importance, scored = _run("synthetic", n=n, seed=seed)

# ---------------- Header metrics ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Events scored", f"{metrics['n_events']:,}")
c2.metric("Out-of-sample MAE", f"${metrics['mae_model']:.0f}",
          help="5-fold cross-validated")
c3.metric("Naive genre-mean MAE", f"${metrics['mae_genre_mean']:.0f}")
c4.metric("Lift vs baseline", f"{metrics['lift_vs_genre_mean_pct']:.0f}%",
          help="MAE improvement over predicting each genre's mean price")
c5.metric("P10–P90 band coverage",
          f"{metrics['band_coverage_pct']:.0f}%",
          help="Empirical coverage of the 80% fair-value band — the "
               "regression analogue of calibration. Should be ≈80%.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Deal finder", "Model quality", "Price drivers", "Walk-forward backtest"])

# ---------------- Tab 1: deal finder ----------------
with tab1:
    liquid = scored[scored["listing_count"] >= min_listings]
    st.subheader("Listed below fair value (potential buys)")
    cols = ["title", "taxonomy", "venue_city", "days_to_event",
            "listing_count", "avg_price", "fair_low", "fair_price",
            "fair_high", "deal_score_pct", "confidence"]
    st.dataframe(liquid[cols].head(15), use_container_width=True, hide_index=True)
    st.subheader("Rich vs fair value (avoid / sell)")
    st.dataframe(liquid[cols].tail(10).iloc[::-1], use_container_width=True,
                 hide_index=True)
    st.caption("deal_score_pct = (fair − observed) / fair × 100; every fair "
               "value is an out-of-sample prediction. `confidence` is high "
               "only when the observed price falls outside the entire "
               "P10–P90 band — a price merely below the point estimate can "
               "just be band noise.")

# ---------------- Tab 2: model quality ----------------
with tab2:
    fig = go.Figure(go.Scattergl(
        x=scored["avg_price"], y=scored["fair_price"], mode="markers",
        marker=dict(size=5, opacity=0.45, color="#1a3d6d"),
        text=scored["title"], name="events"))
    lim = float(max(scored["avg_price"].quantile(0.99),
                    scored["fair_price"].quantile(0.99)))
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                  line=dict(dash="dot", color="#888"))
    fig.update_layout(xaxis_title="observed avg price (USD)",
                      yaxis_title="predicted fair price (USD)",
                      xaxis_range=[0, lim], yaxis_range=[0, lim],
                      height=480, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"R² (out-of-sample) = {metrics['r2_model']:.3f}. Points far "
               "below the diagonal are priced rich; far above, cheap.")

# ---------------- Tab 3: drivers ----------------
with tab3:
    imp = importance.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"],
                           orientation="h", marker_color="#1a3d6d"))
    fig.update_layout(xaxis_title="permutation importance (log-price)",
                      height=420, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Computed on a held-out fold. In the synthetic market the true "
               "drivers are known by construction (popularity, supply, timing) "
               "— the model recovers them, which is the estimator sanity check.")

# ---------------- Tab 4: walk-forward backtest ----------------
with tab4:
    st.markdown(
        "**Does the signal actually pay?** Cross-sectional accuracy is not "
        "enough — a deal score is only real if prices subsequently move "
        "toward fair value. Protocol: weekly panel, expanding-window refit "
        "(each week's model sees only the past), score week *t*, realize "
        "returns over week *t+1*. Synthetic panel with persistent, "
        "mean-reverting mispricing — the harness is what transfers to real "
        "listing history.")

    @st.cache_data(show_spinner="Running walk-forward backtest…")
    def _backtest():
        from ticketsignal.backtest import simulate_panel, walk_forward_backtest
        bt = walk_forward_backtest(simulate_panel())
        return bt.metrics, bt.deciles

    bm, deciles = _backtest()
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Rank IC (weekly mean)", f"{bm['rank_ic_mean']:.3f}",
              help="Spearman corr. of deal score vs next-week return")
    b2.metric("IC t-stat", f"{bm['rank_ic_tstat']:.1f}")
    b3.metric("Top-decile fwd ret", f"{bm['top_decile_fwd_ret_pct']:.1f}%/wk")
    b4.metric("Top − bottom spread", f"{bm['top_minus_bottom_pct']:.1f}%/wk")

    fig = go.Figure(go.Bar(
        x=deciles["bucket"], y=deciles["mean_fwd_ret_pct"],
        marker_color=np.where(deciles["mean_fwd_ret_pct"] >= 0,
                              "#1a3d6d", "#c0392b").tolist()))
    fig.update_layout(xaxis_title="deal-score decile (1 = most overpriced, "
                                  "10 = most underpriced)",
                      yaxis_title="mean next-week return (%)",
                      height=420, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{bm['weeks_tested']} weeks out of sample, "
               f"{bm['n_signal_obs']:,} signal observations. A monotone ramp "
               "is the signature of a working signal; a flat one would say "
               "the model only explains prices, not opportunities.")
