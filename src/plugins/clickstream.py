"""Clickstream plugin: predict session conversion from pre-decision events.

Enhanced feature engineering with richer behavioral signals extracted
from pre-decision events only (page_view, click).
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .base import TrainingPlugin


def _stable_hash_mod(s, mod):
    """Deterministic hash → bucket. Python's built-in hash() is randomized
    per process via PYTHONHASHSEED, so feature values would differ across
    runs of the same code on the same data — poisoning the agent loop."""
    if pd.isna(s):
        return -1
    digest = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod


_PRE_DECISION_EVENT_TYPES = ("page_view", "click")


class ClickstreamPlugin(TrainingPlugin):
    name = "clickstream"
    task = "classification"

    def prepare(self, df: pd.DataFrame):
        # 1. Keep only pre-decision events for feature computation.
        pre = df[df["event_type"].isin(_PRE_DECISION_EVENT_TYPES)].copy()

        # Ensure timestamp is datetime
        pre["timestamp"] = pd.to_datetime(pre["timestamp"])

        # 2. Per-session pre-decision event-type counts.
        type_counts = (
            pre.assign(_one=1)
               .pivot_table(
                   index="session_id",
                   columns="event_type",
                   values="_one",
                   aggfunc="sum",
                   fill_value=0,
               )
        )
        for col in _PRE_DECISION_EVENT_TYPES:
            if col not in type_counts.columns:
                type_counts[col] = 0
        type_counts = type_counts[list(_PRE_DECISION_EVENT_TYPES)].rename(
            columns={"page_view": "n_page_views", "click": "n_clicks"}
        )
        type_counts["n_pre_events"] = type_counts.sum(axis=1)

        # Ratio features: clicks to total pre-events (cohort signal)
        type_counts["click_ratio"] = (
            type_counts["n_clicks"] / (type_counts["n_pre_events"] + 1e-9)
        )
        type_counts["page_view_ratio"] = (
            type_counts["n_page_views"] / (type_counts["n_pre_events"] + 1e-9)
        )
        # Log transforms to handle skew
        type_counts["log_n_clicks"] = np.log1p(type_counts["n_clicks"])
        type_counts["log_n_page_views"] = np.log1p(type_counts["n_page_views"])
        type_counts["log_n_pre_events"] = np.log1p(type_counts["n_pre_events"])

        # 3. Referrer dummies (one-hot for pre-decision events)
        referrer_dummies = (
            pre.groupby(["session_id", "referrer"])
               .size()
               .unstack(fill_value=0)
        )
        referrer_dummies.columns = [f"ref_{c}" for c in referrer_dummies.columns]
        # Normalize referrer counts by n_pre_events
        referrer_totals = referrer_dummies.sum(axis=1).replace(0, 1)
        referrer_ratios = referrer_dummies.div(referrer_totals, axis=0)
        referrer_ratios.columns = [f"{c}_ratio" for c in referrer_dummies.columns]

        # 4. Per-session contextual features (still pre-decision only).
        ctx = pre.groupby("session_id").agg(
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
            n_referrers=("referrer", "nunique"),
            ip_bucket=("ip_bucket", "first"),
            has_user_id=("user_id", lambda s: int(s.notna().any())),
        )
        ctx["pre_duration_s"] = (
            (ctx["last_ts"] - ctx["first_ts"]).dt.total_seconds()
        )

        # Time features from first event
        ctx["hour_of_day"] = ctx["first_ts"].dt.hour
        ctx["day_of_week"] = ctx["first_ts"].dt.dayofweek
        ctx["is_weekend"] = (ctx["day_of_week"] >= 5).astype(int)
        # Hour buckets (morning/afternoon/evening/night)
        ctx["hour_bucket"] = pd.cut(ctx["first_ts"].dt.hour,
                                     bins=[0, 6, 12, 18, 24],
                                     labels=[0, 1, 2, 3],
                                     right=False).astype(float)
        # Log duration
        ctx["log_duration_s"] = np.log1p(ctx["pre_duration_s"])

        ctx = ctx.drop(columns=["first_ts", "last_ts"])

        # 5. Inter-event timing features
        def timing_features(group):
            ts = group["timestamp"].sort_values()
            if len(ts) < 2:
                return pd.Series({
                    "median_gap_s": 0.0,
                    "max_gap_s": 0.0,
                    "min_gap_s": 0.0,
                    "std_gap_s": 0.0,
                    "mean_gap_s": 0.0,
                    "p25_gap_s": 0.0,
                    "p75_gap_s": 0.0,
                    "n_rapid_events": 0.0,
                })
            gaps = ts.diff().dropna().dt.total_seconds()
            return pd.Series({
                "median_gap_s": gaps.median(),
                "max_gap_s": gaps.max(),
                "min_gap_s": gaps.min(),
                "std_gap_s": gaps.std() if len(gaps) > 1 else 0.0,
                "mean_gap_s": gaps.mean(),
                "p25_gap_s": gaps.quantile(0.25),
                "p75_gap_s": gaps.quantile(0.75),
                "n_rapid_events": float((gaps < 5.0).sum()),
            })

        timing = pre.groupby("session_id").apply(timing_features)

        # 6. Device fingerprint hash bucket (loose neighborhood signal)
        device_feat = pre.groupby("session_id").agg(
            device_fp=("device_fingerprint", "first"),
        )
        device_feat["device_bucket_32"] = device_feat["device_fp"].apply(
            lambda x: _stable_hash_mod(x, 32)
        )
        device_feat["device_bucket_64"] = device_feat["device_fp"].apply(
            lambda x: _stable_hash_mod(x, 64)
        )
        device_feat["device_bucket_16"] = device_feat["device_fp"].apply(
            lambda x: _stable_hash_mod(x, 16)
        )
        device_feat = device_feat.drop(columns=["device_fp"])

        # 7. IP bucket features — count sessions sharing same ip_bucket
        ip_session_counts = (
            pre.groupby("session_id")["ip_bucket"].first()
               .reset_index()
               .rename(columns={"ip_bucket": "ip_bkt"})
        )
        ip_freq = ip_session_counts["ip_bkt"].value_counts().to_dict()
        ip_session_counts["ip_bucket_freq"] = ip_session_counts["ip_bkt"].map(ip_freq)
        ip_session_counts = ip_session_counts.set_index("session_id")[["ip_bucket_freq"]]

        # 8. Referrer sequence features — first and last referrer
        referrer_seq = pre.sort_values("timestamp").groupby("session_id")["referrer"].agg(
            first_referrer="first",
            last_referrer="last",
        )
        # Encode referrers as integers
        referrer_cats = ["google", "direct", "facebook", "email", "affiliate", "organic"]
        referrer_map = {r: i for i, r in enumerate(referrer_cats)}
        referrer_seq["first_referrer_enc"] = referrer_seq["first_referrer"].map(referrer_map).fillna(-1)
        referrer_seq["last_referrer_enc"] = referrer_seq["last_referrer"].map(referrer_map).fillna(-1)
        referrer_seq["referrer_changed"] = (
            referrer_seq["first_referrer"] != referrer_seq["last_referrer"]
        ).astype(int)
        referrer_seq = referrer_seq.drop(columns=["first_referrer", "last_referrer"])

        # 9. Click/page_view alternation pattern
        def sequence_features(group):
            types = group.sort_values("timestamp")["event_type"].values
            n = len(types)
            if n < 2:
                return pd.Series({
                    "click_after_pageview": 0.0,
                    "pageview_after_click": 0.0,
                    "alternation_rate": 0.0,
                })
            transitions = [(types[i], types[i+1]) for i in range(n-1)]
            cap = sum(1 for a, b in transitions if a == "page_view" and b == "click")
            pac = sum(1 for a, b in transitions if a == "click" and b == "page_view")
            alternations = sum(1 for a, b in transitions if a != b)
            return pd.Series({
                "click_after_pageview": float(cap),
                "pageview_after_click": float(pac),
                "alternation_rate": alternations / max(len(transitions), 1),
            })

        seq_feats = pre.groupby("session_id").apply(sequence_features)

        # 10. Per-session target
        targets = (
            df.groupby("session_id")["session_converted"].first().astype(int)
        )

        # 11. Combine all features. Inner join keeps sessions with pre-decision events.
        features = (
            type_counts
            .join(ctx, how="inner")
            .join(timing, how="left")
            .join(referrer_dummies, how="left")
            .join(referrer_ratios, how="left")
            .join(device_feat, how="left")
            .join(ip_session_counts, how="left")
            .join(referrer_seq, how="left")
            .join(seq_feats, how="left")
        )
        features = features.fillna(0)

        # 12. Interaction features
        features["duration_per_event"] = (
            features["pre_duration_s"] / (features["n_pre_events"] + 1e-9)
        )
        features["clicks_per_minute"] = (
            features["n_clicks"] / (features["pre_duration_s"] / 60.0 + 1e-9)
        )
        features["page_views_per_minute"] = (
            features["n_page_views"] / (features["pre_duration_s"] / 60.0 + 1e-9)
        )
        # Click ratio x duration interaction
        features["click_ratio_x_duration"] = (
            features["click_ratio"] * features["log_duration_s"]
        )
        # n_clicks x referrer signal
        if "ref_google_ratio" in features.columns:
            features["clicks_x_google"] = features["n_clicks"] * features["ref_google_ratio"]
        if "ref_email_ratio" in features.columns:
            features["clicks_x_email"] = features["n_clicks"] * features["ref_email_ratio"]
        # Timing entropy proxy: std/mean ratio (coefficient of variation)
        features["gap_cv"] = (
            features["std_gap_s"] / (features["mean_gap_s"] + 1e-9)
        )
        # Inter-quartile range of gaps
        features["gap_iqr"] = features["p75_gap_s"] - features["p25_gap_s"]

        y = targets.loc[features.index]

        return features.reset_index(drop=True), y.reset_index(drop=True)

    def evaluate(self, model, X_test, y_true):
        proba = model.predict_proba(X_test)[:, 1]
        return "validation_auc", float(roc_auc_score(y_true, proba))

    def build_model(self, params: dict):
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 600)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.02)),
            min_samples_leaf=int(params.get("min_samples_leaf", 10)),
            l2_regularization=float(params.get("l2_regularization", 0.05)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            class_weight="balanced",
            random_state=42,
        )

    def extra_config(self, model, X: pd.DataFrame) -> dict:
        # ~6% positive class — default 0.5 threshold is too conservative
        # and produces near-zero recall. 0.15 gives a more balanced
        # precision/recall split. Tune per deployment context by editing
        # `prediction_threshold` in the bundle's config.json (no retrain
        # needed) — the wrapper / loader reads it at inference time.
        return {"prediction_threshold": 0.15}