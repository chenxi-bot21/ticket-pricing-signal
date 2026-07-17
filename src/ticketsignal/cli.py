"""CLI: run the full pipeline and write artifacts.

    python -m ticketsignal.cli --source synthetic
    python -m ticketsignal.cli --source seatgeek --client-id YOUR_ID
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import fetch_seatgeek_events, synthetic_events
from .features import build_features
from .model import train_and_score


def main() -> None:
    ap = argparse.ArgumentParser(description="Ticket fair-value signal")
    ap.add_argument("--source", choices=["synthetic", "seatgeek", "csv"],
                    default="synthetic")
    ap.add_argument("--client-id", default=None,
                    help="SeatGeek client id (free at seatgeek.com/build)")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--n", type=int, default=1500,
                    help="synthetic events to generate")
    ap.add_argument("--from-csv", default=None,
                    help="re-run offline from a cached raw pull")
    ap.add_argument("--save-raw", default=None,
                    help="cache the raw pull to CSV (reproducible offline)")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--backtest", action="store_true",
                    help="also run the walk-forward panel backtest")
    args = ap.parse_args()

    if args.source == "csv":
        if not args.from_csv:
            ap.error("--from-csv is required with --source csv")
        import pandas as pd
        df = pd.read_csv(args.from_csv)
    elif args.source == "seatgeek":
        if not args.client_id:
            ap.error("--client-id is required with --source seatgeek")
        df = fetch_seatgeek_events(args.client_id, pages=args.pages)
    else:
        df = synthetic_events(n=args.n)

    if args.save_raw:
        Path(args.save_raw).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_raw, index=False)
        print(f"raw pull cached -> {args.save_raw}  (rows={len(df)})")

    feats = build_features(df)
    report = train_and_score(feats)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.scored.to_csv(out / "events_scored.csv", index=False)
    report.importance.to_csv(out / "importance.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report.metrics, indent=2))

    m = report.metrics
    print(f"events scored: {m['n_events']}")
    print(f"MAE  model {m['mae_model']:.2f} | genre-mean baseline "
          f"{m['mae_genre_mean']:.2f} | global-mean {m['mae_global_mean']:.2f}")
    print(f"R2 {m['r2_model']:.3f} | lift vs genre-mean "
          f"{m['lift_vs_genre_mean_pct']:.1f}%")
    print(f"P10-P90 band coverage: {m['band_coverage_pct']:.1f}% "
          f"(nominal {m['band_nominal_pct']:.0f}%)")
    print("\ntop importance:")
    print(report.importance.head(6).to_string(index=False))
    print("\ntop 5 potential buys (listed below fair value):")
    cols = ["title", "taxonomy", "days_to_event", "avg_price",
            "fair_price", "fair_low", "deal_score_pct", "confidence"]
    print(report.scored[cols].head(5).to_string(index=False))
    if args.backtest:
        from .backtest import simulate_panel, walk_forward_backtest
        print("\n=== walk-forward backtest (synthetic panel) ===")
        panel = simulate_panel()
        bt = walk_forward_backtest(panel)
        bm = bt.metrics
        print(f"weeks tested: {bm['weeks_tested']} | signal obs: "
              f"{bm['n_signal_obs']}")
        print(f"rank IC {bm['rank_ic_mean']:.3f} (t={bm['rank_ic_tstat']:.1f}) | "
              f"top decile fwd ret {bm['top_decile_fwd_ret_pct']:.2f}% vs bottom "
              f"{bm['bottom_decile_fwd_ret_pct']:.2f}% | spread "
              f"{bm['top_minus_bottom_pct']:.2f}%")
        print(bt.deciles.to_string(index=False))
        bt.deciles.to_csv(out / "backtest_deciles.csv", index=False)
        (out / "backtest_metrics.json").write_text(
            json.dumps(bm, indent=2))

    print(f"\nartifacts -> {out.resolve()}")


if __name__ == "__main__":
    main()
