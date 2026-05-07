"""Write the iris dataset to data/iris.csv so SageMaker can mount it as a channel."""
import os
import shutil
from sklearn.datasets import load_iris

if os.path.isdir("data"):
    shutil.rmtree("data")
os.makedirs("data")

df = load_iris(as_frame=True).frame
df.to_csv("data/iris.csv", index=False)
print(f"wrote data/iris.csv ({len(df)} rows)")
