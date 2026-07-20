"""Tests: data contract, leakage guard, model beats naive baselines, deal math,
and the walk-forward backtest actually validates the signal."""
import unittest

import numpy as np

from ticketsignal.backtest import simulate_panel, walk_forward_backtest
from ticketsignal.data import synthetic_events
from ticketsignal.features import CATEGORICAL, NUMERIC, TARGET, build_features
from ticketsignal.model import train_and_score


class TestData(unittest.TestCase):
    def test_synthetic_contract_and_reproducibility(self):
        df = synthetic_events(n=500, seed=7)
        self.assertEqual(len(df), 500)
        for col in ["event_id", "taxonomy", "days_to_event", "weekend",
                    "event_score", "performer_score", "listing_count", TARGET]:
            self.assertIn(col, df.columns)
        self.assertTrue((df[TARGET] > 0).all())
        df2 = synthetic_events(n=500, seed=7)
        self.assertTrue(df[TARGET].equals(df2[TARGET]))  # seeded


class TestFeatures(unittest.TestCase):
    def test_features_complete_and_target_free(self):
        feats = build_features(synthetic_events(n=400))
        self.assertFalse(feats[NUMERIC].isna().any().any())
        # leakage guard: the target never appears in the model inputs
        self.assertNotIn(TARGET, NUMERIC + CATEGORICAL)

    def test_missing_column_raises(self):
        df = synthetic_events(n=50).drop(columns=["listing_count"])
        with self.assertRaises(ValueError):
            build_features(df)


class TestModel(unittest.TestCase):
    def test_model_beats_baselines_and_deal_math(self):
        feats = build_features(synthetic_events(n=1200))
        rep = train_and_score(feats)
        m = rep.metrics
        # honest bar: out-of-sample model must beat the per-genre mean
        self.assertLess(m["mae_model"], m["mae_genre_mean"])
        self.assertGreater(m["r2_model"], 0.5)
        # deal score arithmetic
        s = rep.scored.iloc[0]
        expected = 100 * (s["fair_price"] - s[TARGET]) / s["fair_price"]
        self.assertAlmostEqual(s["deal_score_pct"], round(expected, 1), places=1)
        # known drivers should rank highly in importance
        top = set(rep.importance.head(4)["feature"])
        self.assertTrue({"event_score", "log_listings"} & top)
        # uncertainty band: sane ordering and honest coverage (nominal 80%)
        s = rep.scored
        self.assertGreater((s["fair_low"] <= s["fair_high"]).mean(), 0.99)
        self.assertTrue(55.0 <= m["band_coverage_pct"] <= 95.0)
        self.assertIn("confidence", s.columns)


class TestTicketmasterParser(unittest.TestCase):
    def _event(self, **over):
        from datetime import datetime, timedelta, timezone
        start = (datetime.now(timezone.utc) + timedelta(days=30))
        ev = {
            "id": "tm1", "name": "Test Show",
            "priceRanges": [{"type": "standard", "min": 40.0, "max": 80.0}],
            "dates": {"start": {
                "dateTime": start.strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "classifications": [{"genre": {"name": "Rock"},
                                 "segment": {"name": "Music"}}],
            "_embedded": {
                "venues": [{"city": {"name": "Boston"},
                            "upcomingEvents": {"_total": 55}}],
                "attractions": [{"upcomingEvents": {"_total": 12}}]},
        }
        ev.update(over)
        return ev

    def test_parse_and_finalize_contract(self):
        from datetime import datetime, timezone
        from ticketsignal.ticketmaster import _finalize, _parse_event
        now = datetime.now(timezone.utc)
        row = _parse_event(self._event(), now)
        self.assertEqual(row["avg_price"], 60.0)      # (40+80)/2
        self.assertEqual(row["taxonomy"], "rock")
        # events without a standard price range are skipped
        self.assertIsNone(_parse_event(self._event(priceRanges=[]), now))
        df = _finalize([_parse_event(self._event(id=f"tm{i}"), now)
                        for i in range(10)])
        for col in ["event_score", "performer_score", "venue_tier", TARGET]:
            self.assertIn(col, df.columns)
        self.assertTrue(df["performer_score"].between(0, 1).all())

    def test_venue_tier_is_leave_one_out(self):
        import pandas as pd
        from ticketsignal.ticketmaster import venue_tier
        # same venue, one cheap + one expensive event: each event's tier is
        # driven by the OTHER's price, so the cheap event gets the higher tier
        df = pd.DataFrame({
            "venue_id": ["v1", "v1", "v2"],
            "avg_price": [10.0, 1000.0, 50.0]})
        tier = venue_tier(df)
        self.assertGreater(tier.iloc[0], tier.iloc[1])
        self.assertEqual(tier.iloc[2], 0.5)      # single-event venue: neutral


class TestHighCardinality(unittest.TestCase):
    def test_many_categories_still_fit(self):
        # real-world pulls (e.g. Ticketmaster) carry dozens of genres/cities;
        # the one-hot block must stay dense or the GBM rejects it
        df = synthetic_events(n=400, seed=3)
        rng = np.random.default_rng(3)
        df["taxonomy"] = rng.choice([f"genre_{i}" for i in range(40)], len(df))
        df["venue_city"] = rng.choice([f"city_{i}" for i in range(120)], len(df))
        rep = train_and_score(build_features(df))
        self.assertEqual(len(rep.scored), len(df))


class TestBacktest(unittest.TestCase):
    def test_walk_forward_signal_pays(self):
        panel = simulate_panel(n_events=300, n_weeks=18, seed=11)
        # panel honours the snapshot contract week by week
        self.assertEqual(
            panel.groupby(["week", "event_id"]).size().max(), 1)
        bt = walk_forward_backtest(panel, min_train_weeks=6)
        m = bt.metrics
        # a real signal: positive rank IC and top decile beats bottom
        self.assertGreater(m["rank_ic_mean"], 0.05)
        self.assertGreater(m["top_minus_bottom_pct"], 0.0)
        self.assertGreaterEqual(m["weeks_tested"], 8)


if __name__ == "__main__":
    unittest.main()
