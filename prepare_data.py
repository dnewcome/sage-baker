"""Write the iris dataset to data/iris.csv so SageMaker can mount it as a channel."""
import os
import pandas as pd
from sklearn.datasets import load_iris

os.makedirs("data", exist_ok=True)
iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={"target": "target"})
df.to_csv("data/iris.csv", index=False)
print(f"wrote data/iris.csv ({len(df)} rows)")
