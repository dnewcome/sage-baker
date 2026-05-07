from feast import Entity, ValueType


# An entity is *what* features are about. For sonar, each row in the dataset
# is a single sonar return — `signal_id` is the join key feature views use
# to attach feature values.
sonar_signal = Entity(
    name="sonar_signal",
    join_keys=["signal_id"],
    value_type=ValueType.INT64,
    description="A single underwater sonar return signal",
)
