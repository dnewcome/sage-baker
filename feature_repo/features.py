from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float32

from entities import sonar_signal


# Where the feature values live. FileSource reads parquet by default — to
# point at S3 in a SageMaker setup, just change `path` to s3://bucket/key.
sonar_source = FileSource(
    path="data/sonar_features.parquet",
    timestamp_field="event_timestamp",
)


# A feature view is a logical group of features keyed by an entity. Here:
# the 60 sonar frequency bands per signal. Same view can be used at
# training time (point-in-time historical join) and at serving time
# (online lookup by signal_id).
sonar_bands = FeatureView(
    name="sonar_bands",
    entities=[sonar_signal],
    ttl=timedelta(days=365),
    schema=[Field(name=f"f{i}", dtype=Float32) for i in range(60)],
    source=sonar_source,
    online=True,
)
