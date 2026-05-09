"""fuzzy_clickstream: clickstream events with fuzzy identity + conversion labels.

This single dataset supports three distinct ML problems, which is why
it's the canonical first scenario:

  1. Record linkage / similarity. Many events have user_id=None
     (logged-out browsing). Model task: given two events, decide if
     they came from the same true user. Ground truth in
     ground_truth.parquet's `true_user_id` column.

  2. Conversion classification. Per-session label `session_converted`
     in the training frame. Standard imbalanced binary problem; the
     positive rate is ~5% by default. Cohort-driven heterogeneity
     means simple "last-touch" features don't dominate.

  3. Attribution (bonus). Each conversion's preceding event chain is
     recorded in ground_truth (`is_converted_session` + the natural
     ordering of session_id / timestamp). A model can compete against
     a last-click baseline.

Realism knobs
-------------
- IP-bucket sharing (default on): N users share each /24-ish bucket,
  so IP alone is a noisy similarity signal.
- Device fingerprint stability (default high): same user → same
  fingerprint within the simulation window.
- Logged-out rate per user (varies in [0.1, 0.9]): user_id is missing
  on a per-session basis, controlled by the user's base_login_rate.
- Cohort-driven conversion lift: `intender` cohort converts ~2.5×
  more than baseline.
- `easy_mode=True` flag: forces every session to be logged in and
  fingerprints to be unique-per-user. Useful for sanity-check
  baselines (any model should hit AUC ~ 1 on `same_user` prediction
  with logged-in events).

TODO
----
- Time-of-day periodicity in event volume.
- Drift over the window (new users, campaign refresh, conversion-rate shift).
- Fraud subpopulation: a few users with bot-like patterns.
- Cross-device journeys: multi-fingerprint same-user.
- Multi-touch attribution windows: clicks N days before conversion.
- Refunds / chargebacks (label noise on the positive class).
"""
import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..base import Scenario, SimulationResult
from ..events import EVENT_TYPES_BROWSE, REFERRERS
from ..population import make_population


class FuzzyClickstreamScenario(Scenario):
    name = "fuzzy_clickstream"
    description = (
        "Clickstream events with fuzzy identity (sometimes-null user_id) "
        "and conversion labels. Supports record linkage, conversion "
        "prediction, and attribution problems."
    )
    default_params = {
        "n_users": 500,
        "days": 14,
        "sessions_per_user": 4,
        "base_conversion_rate": 0.05,
        # Default 0.0 — every user is anonymous (user_id always None).
        # Bump up to seed identified users; e.g. 0.1 = 10% of the
        # population can sometimes be logged in, simulating a small
        # logged-in cohort within an otherwise anonymous traffic mix.
        "identified_user_fraction": 0.0,
        "easy_mode": False,
    }

    def generate(self, seed: int = 42, **params) -> SimulationResult:
        p = {**self.default_params, **params}
        rng = random.Random(seed)
        identified_fraction = (
            1.0 if p["easy_mode"] else float(p["identified_user_fraction"])
        )
        users = make_population(rng, p["n_users"], identified_fraction=identified_fraction)

        events: list[dict] = []
        ground_truth: list[dict] = []

        sim_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        event_id = 0
        session_id = 0

        for user in users:
            # Sessions per user is a rough average; jitter slightly.
            n_sessions = max(1, int(rng.gauss(p["sessions_per_user"], 1)))
            for _ in range(n_sessions):
                session_id += 1
                session_start = sim_start + timedelta(
                    days=rng.uniform(0, p["days"]),
                    seconds=rng.randint(0, 86400),
                )

                # In easy_mode, every session is logged in & fingerprints
                # are unique-per-user — a sanity-check baseline.
                logged_in = (
                    True if p["easy_mode"]
                    else rng.random() < user.base_login_rate
                )

                # Will this session convert? Cohort propensity * base rate.
                p_convert = min(0.95, p["base_conversion_rate"] * user.convert_propensity)
                will_convert = rng.random() < p_convert

                n_events = rng.randint(1, 8)
                for i in range(n_events):
                    ts = session_start + timedelta(
                        seconds=i * rng.randint(5, 60) + rng.randint(0, 30)
                    )

                    # Funnel-aware event type: late events in a converting
                    # session step toward the conversion.
                    if will_convert and i == n_events - 1:
                        etype = "conversion"
                        value = round(rng.uniform(10.0, 200.0), 2)
                    elif will_convert and i >= n_events - 3:
                        etype = rng.choice(("add_to_cart", "checkout"))
                        value = 0.0
                    else:
                        etype = rng.choice(EVENT_TYPES_BROWSE)
                        value = 0.0

                    events.append({
                        "event_id": event_id,
                        "timestamp": ts,
                        "session_id": session_id,
                        "user_id": user.user_id if logged_in else None,
                        "device_fingerprint": user.device_fingerprint,
                        "ip_bucket": user.ip_bucket,
                        "event_type": etype,
                        "referrer": rng.choice(REFERRERS),
                        "value": value,
                    })
                    ground_truth.append({
                        "event_id": event_id,
                        "true_user_id": user.user_id,
                        "true_cohort": user.cohort,
                        "true_session_id": session_id,
                        "true_logged_in": logged_in,
                        "is_converted_session": will_convert,
                    })
                    event_id += 1

        training_df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
        gt_df = pd.DataFrame(ground_truth)

        # Session-level conversion label, broadcast back to event rows so
        # the existing supervised trainers can use it as `target` directly.
        session_label = (
            training_df.assign(_is_conv=training_df["event_type"].eq("conversion"))
            .groupby("session_id")["_is_conv"].any()
        )
        training_df["session_converted"] = (
            training_df["session_id"].map(session_label).astype(int)
        )

        lineage = self.make_lineage(self.name, seed, p, training_df)
        return SimulationResult(training=training_df, ground_truth=gt_df, lineage=lineage)
