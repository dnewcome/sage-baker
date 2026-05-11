# sagebaker — embedding serving image
#
# Dependency family: PyTorch + sentence-transformers + FAISS + spaCy.
# Suitable for: nearest-neighbour product recommenders, semantic search,
# and any model requiring dense vector embeddings at inference time.
#
# Runtime configuration (ECS env vars):
#   PLUGIN_NAME         name of the plugin to serve (required)
#   MODEL_S3_URI        s3://bucket/prefix/ to download bundle from (required)
#   PLUGIN_DIR          path to plugin .py files if not baked into image (optional)
#   PORT                serving port (default: 8080)
#   TRANSFORMERS_OFFLINE set to 1 to disable HuggingFace hub lookups at runtime
#
# Build args:
#   SAGEBAKER_VERSION
#   FAISS_PKG          faiss-cpu or faiss-gpu (default: faiss-cpu)
#   SPACY_MODEL        spaCy model to download (default: en_core_web_sm)

ARG PYTORCH_DLC=763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.3.0-cpu-py311
FROM ${PYTORCH_DLC}

ARG SAGEBAKER_VERSION=""
ARG FAISS_PKG=faiss-cpu
ARG SPACY_MODEL=en_core_web_sm

RUN pip install --no-cache-dir \
    "sagebaker${SAGEBAKER_VERSION:+==$SAGEBAKER_VERSION}" \
    sentence-transformers \
    "${FAISS_PKG}" \
    spacy \
    && python -m spacy download ${SPACY_MODEL}

ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "-m", "serve"]
