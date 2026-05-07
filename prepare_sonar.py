"""Fetch the Connectionist Bench (Sonar Rocks vs Mines) dataset.

Same dataset as the Kaggle "Underwater Sonar Signals" listing, originally from
Gorman & Sejnowski (1988). Pulled from a public mirror so no Kaggle auth is
needed. 208 instances, 60 numeric sonar-frequency features, binary label
(R = rock, M = mine).
"""
import os
import shutil
import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "sonar.csv")

if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

cols = [f"f{i}" for i in range(60)] + ["target"]
df = pd.read_csv(URL, header=None, names=cols)
df["target"] = df["target"].map({"R": 0, "M": 1})
df.to_csv(OUT_FILE, index=False)

print(f"wrote {OUT_FILE} ({len(df)} rows, "
      f"{(df['target'] == 0).sum()} rocks, {(df['target'] == 1).sum()} mines)")
