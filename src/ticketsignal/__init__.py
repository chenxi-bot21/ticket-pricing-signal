"""ticketsignal — leakage-safe fair-value pricing signal for the secondary ticket market.

Proof of concept: estimate a fair market price for live-event tickets from
listing-time information only, then flag listings that trade above/below fair
value (a "deal score"). The modeling discipline is the point:

- features restricted to what is knowable at listing time (no look-ahead);
- honest out-of-sample evaluation against naive baselines;
- permutation importance for explainability.

Data sources: the public SeatGeek API (free client id) or a seeded synthetic
market with known price drivers, so the pipeline runs and is testable offline.
"""

from .data import fetch_seatgeek_events, synthetic_events
from .features import build_features
from .model import train_and_score

__version__ = "0.1.0"
__all__ = ["fetch_seatgeek_events", "synthetic_events", "build_features",
           "train_and_score", "__version__"]
