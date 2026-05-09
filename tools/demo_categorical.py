"""Demonstrate the "new enum value at inference" bug and three responses.

Synthesizes a tiny dataset with one categorical feature (`browser`),
trains three classifiers, then deliberately injects a new browser
value the trainer never saw. Each path reacts differently:

  1. sklearn + OneHotEncoder (default `handle_unknown="error"`)
        → crashes at predict time. This is the typical production bug.
  2. sklearn + OneHotEncoder(handle_unknown="ignore")
        → silently drops the unknown to all-zeros. No crash; you lose
          whatever signal "edge" carried.
  3. LightGBM + OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1) with categorical_feature
        → maps unseen values to -1, which LightGBM treats as a category
          group of its own. No crash, no signal loss.

Run:  python demo_categorical.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import lightgbm as lgb


def make_data():
    np.random.seed(42)
    N = 1000
    browsers = np.random.choice(["chrome", "firefox", "safari"],
                                size=N, p=[0.6, 0.3, 0.1])
    x = np.random.rand(N) * 10
    y = ((browsers == "firefox") | (x > 7)).astype(int)
    df = pd.DataFrame({"browser": browsers, "x": x, "target": y})
    return train_test_split(df, test_size=0.2, random_state=42)


def inject_unseen(test_df):
    """Replace 5 rows' browser value with 'edge' — never seen in training."""
    out = test_df.copy()
    out.loc[out.index[:5], "browser"] = "edge"
    return out


def try_path(name, fn):
    print(f"\n--- {name} ---")
    try:
        preds = fn()
        print(f"OK   predictions: {preds.tolist()}")
    except Exception as e:
        print(f"CRASH {type(e).__name__}: {str(e)[:200]}")


def path1_sklearn_default(train_df, test_df):
    """OneHotEncoder default = handle_unknown='error'. Production-typical bug."""
    pipe = Pipeline([
        ("enc", ColumnTransformer([
            ("browser", OneHotEncoder(), ["browser"]),
        ], remainder="passthrough")),
        ("rf", RandomForestClassifier(random_state=42)),
    ])
    pipe.fit(train_df[["browser", "x"]], train_df["target"])
    return pipe.predict(test_df[["browser", "x"]].head(5))


def path2_sklearn_ignore(train_df, test_df):
    """OneHotEncoder(handle_unknown='ignore'). Doesn't crash; loses signal."""
    pipe = Pipeline([
        ("enc", ColumnTransformer([
            ("browser", OneHotEncoder(handle_unknown="ignore"), ["browser"]),
        ], remainder="passthrough")),
        ("rf", RandomForestClassifier(random_state=42)),
    ])
    pipe.fit(train_df[["browser", "x"]], train_df["target"])
    return pipe.predict(test_df[["browser", "x"]].head(5))


def path3_lightgbm_native(train_df, test_df):
    """OrdinalEncoder maps unseen → -1, LightGBM treats that as its own category."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = train_df[["browser", "x"]].copy()
    X_test = test_df[["browser", "x"]].copy()
    X_train["browser"] = enc.fit_transform(X_train[["browser"]])
    X_test["browser"] = enc.transform(X_test[["browser"]])

    clf = lgb.LGBMClassifier(verbose=-1)
    clf.fit(X_train, train_df["target"], categorical_feature=["browser"])
    return clf.predict(X_test.head(5))


if __name__ == "__main__":
    train_df, test_df = make_data()
    test_df = inject_unseen(test_df)

    print(f"trained on browsers: {sorted(train_df['browser'].unique())}")
    print(f"5 inference rows now have unseen value 'edge'")

    try_path("Path 1: sklearn OneHotEncoder default", lambda: path1_sklearn_default(train_df, test_df))
    try_path("Path 2: sklearn OneHotEncoder(handle_unknown='ignore')", lambda: path2_sklearn_ignore(train_df, test_df))
    try_path("Path 3: LightGBM + OrdinalEncoder(unknown_value=-1)", lambda: path3_lightgbm_native(train_df, test_df))
