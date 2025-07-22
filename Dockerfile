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
    pkgconfig

# Install python packages globally
RUN pip install --no-cache-dir numpy pyarrow pytest

# --- Stage 2: The Final Production Image ---
# Start from a minimal, clean alpine image
FROM python:3.11-alpine

# Copy the installed python packages from the builder's global site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy your application code and other necessary files
COPY src/VERSION /VERSION
COPY . /app
WORKDIR /app

# Default command
ENTRYPOINT ["python", "src/app.py"]

CMD ["classify"]