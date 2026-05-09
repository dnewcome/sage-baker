"""Session + Event primitives.

A Session is a User's contiguous interaction with the property over a
short time window. An Event is a single observed action within a
session (page_view, click, add_to_cart, conversion, etc.).

This module defines the *shape* of events. Scenarios assemble them into
a particular distribution (funnels, drift, fraud patterns).

TODO
----
- Time-correlation primitives (events from the same true user tend to
  be temporally clustered; record-linkage models exploit this).
- Bot / fraud event generators (low-time-between-events, distribution
  outliers, geographic mismatches).
- Cross-device journeys.
"""
import dataclasses
from datetime import datetime
from typing import Optional


# Funnel ordering matters — converting sessions tend to step through
# late-stage events near the end.
EVENT_TYPES_FUNNEL = ("page_view", "click", "add_to_cart", "checkout", "conversion")
EVENT_TYPES_BROWSE = ("page_view", "click")
REFERRERS = ("google", "direct", "facebook", "email", "affiliate", "organic")


@dataclasses.dataclass
class Event:
    """The full event shape — scenarios decide which fields land in the
    training frame vs ground truth."""
    event_id: int
    timestamp: datetime
    session_id: int
    # `user_id` is what the model sees. It can be None (logged out).
    user_id: Optional[int]
    # `true_user_id` is ground truth — never in the training frame.
    true_user_id: int
    device_fingerprint: str
    ip_bucket: int
    event_type: str
    referrer: str
    value: float           # 0.0 except on conversion events
    # Ground-truth-only fields:
    cohort: str
    is_session_converted: bool
