# --- Stage 1: The Builder ---
# Use a full-featured image to build our dependencies
FROM python:3.11-alpine AS builder

# Install system dependencies needed for building
RUN apk add --no-cache \
    build-base \
    linux-headers \
    musl-dev \
    gcc \
    g++ \
    gfortran \
    openblas-dev \
    libffi-dev \
    cmake \
    pkgconfig \
    curl-dev \
    openssl-dev

ENV PYARROW_WITH_HTTP=ON

# Install python packages globally
RUN pip install --no-cache-dir --retries 5 --timeout 120 numpy pyarrow requests pytest pytest-dotenv

# --- Stage 2: The Final Production Image ---
# Start from a minimal, clean alpine image
FROM python:3.11-alpine

RUN apk add --no-cache curl openssl

ENV EMBEDDINGS_CLASSIFIER_INPUT=/mnt/input \
    EMBEDDINGS_CLASSIFIER_OUTPUT=/mnt/output \
    EMBEDDINGS_CLASSIFIER_CONFIG=/mnt/config/config.json

# Copy the installed python packages from the builder's global site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy your application code and other necessary files
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY src/VERSION /VERSION
COPY src /app/src
WORKDIR /app

RUN pip install --no-cache-dir --no-deps -e .

# Default command
ENTRYPOINT ["python", "-m", "embeddings_classifier.app"]

CMD ["classify"]