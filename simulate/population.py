"""User population primitives for clickstream-style scenarios.

A Population is a fixed set of synthetic users, each with stable
attributes (cohort, device fingerprint, base IP bucket, login propensity)
that scenarios use to generate events. The user objects are ground truth
— scenarios decide which fields the model gets to see.

Design notes
------------
- Cohorts encode behavior patterns: a `bargain_hunter` shops a lot but
  rarely converts; an `intender` converts in fewer sessions; a
  `loyalist` returns repeatedly.
- IP buckets are smaller than the user count, so multiple users share
  the same bucket — the source of similarity ambiguity that record-
  linkage models have to resolve.
- `device_fingerprint` is stable per user but the namespace is wide,
  so collisions are rare. Realism extension: simulate device updates,
  shared family devices, browser fingerprint drift.

TODO
----
- Multi-device users (same true_user_id, different fingerprints).
- Time-of-day cohort variation.
- Drift over the simulation window (cohorts shift, new users arrive).
"""
import dataclasses
import random


COHORTS = ("browser", "intender", "bargain_hunter", "loyalist")


@dataclasses.dataclass
class User:
    user_id: int
    cohort: str
    device_fingerprint: str
    ip_bucket: int
    base_login_rate: float       # probability a session has user_id recorded
    convert_propensity: float    # multiplier on base conversion rate


def _propensity_for_cohort(cohort: str, rng: random.Random) -> float:
    return {
        "browser": rng.uniform(0.2, 0.6),
        "intender": rng.uniform(1.5, 3.0),
        "bargain_hunter": rng.uniform(0.5, 1.0),
        "loyalist": rng.uniform(1.0, 2.0),
    }[cohort]


def make_population(
    rng: random.Random,
    n_users: int = 500,
    identified_fraction: float = 0.0,
) -> list[User]:
    """Generate a population of n_users with stable identity signals.

    IP-bucket sharing is the key knob for similarity ambiguity: with
    n_buckets ≈ n_users / 3 (the default), about 2/3 of users share a
    bucket with at least one other user.

    `identified_fraction` controls how many users can EVER be
    identified (i.e. have user_id recorded on at least some of their
    sessions). Default 0.0 means everyone is anonymous — the realistic
    case for most user populations. For identifiable users, the
    per-session login probability is drawn from [0.1, 0.9].
    """
    n_buckets = max(1, n_users // 3)
    users = []
    for i in range(n_users):
        cohort = rng.choice(COHORTS)
        is_identifiable = rng.random() < identified_fraction
        users.append(User(
            user_id=i,
            cohort=cohort,
            device_fingerprint=f"fp_{rng.randint(0, n_users * 2):08x}",
            ip_bucket=rng.randint(0, n_buckets - 1),
            # Anonymous users never log in; identifiable ones do so at a
            # user-specific rate. Setting base_login_rate=0 here makes
            # the scenario code branch-free: it just samples this rate.
            base_login_rate=rng.uniform(0.1, 0.9) if is_identifiable else 0.0,
            convert_propensity=_propensity_for_cohort(cohort, rng),
        ))
    return users
