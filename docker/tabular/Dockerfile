# sagebaker — tabular serving image
#
# Dependency family: scikit-learn + LightGBM + imbalanced-learn.
# Suitable for: classification and regression models trained with LightGBM,
# sklearn pipelines, or any combination thereof.
#
# Runtime configuration (ECS env vars):
#   PLUGIN_NAME       name of the plugin to serve (required)
#   MODEL_S3_URI      s3://bucket/prefix/ to download bundle from (required)
#   PLUGIN_DIR        path to plugin .py files if not baked into image (optional)
#   PORT              serving port (default: 8080)
#
# Build args:
#   SAGEBAKER_VERSION  pip version spec for sagebaker (default: installs latest)
#   LIGHTGBM_VERSION
#   IMBALANCED_VERSION

ARG SKLEARN_DLC=763104351884.dkr.ecr.us-west-2.amazonaws.com/sklearn:1.2-1-cpu-py3
FROM ${SKLEARN_DLC}

ARG SAGEBAKER_VERSION=""
ARG LIGHTGBM_VERSION=""
ARG IMBALANCED_VERSION=""

RUN pip install --no-cache-dir \
    "sagebaker${SAGEBAKER_VERSION:+==$SAGEBAKER_VERSION}" \
    "lightgbm${LIGHTGBM_VERSION:+==$LIGHTGBM_VERSION}" \
    "imbalanced-learn${IMBALANCED_VERSION:+==$IMBALANCED_VERSION}"

# Default entrypoint: sagebaker's HTTP serving harness.
# Override PLUGIN_DIR at runtime to inject plugin code without rebuilding.
CMD ["python", "-m", "serve"]
