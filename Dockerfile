FROM python:3.11-alpine

# Install system dependencies needed for building numpy and pyarrow
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

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install numpy first (pyarrow depends on it)
RUN pip install --no-cache-dir numpy

# Install pyarrow
RUN pip install --no-cache-dir pyarrow

# Optional: Install pandas if you need it
# RUN pip install --no-cache-dir pandas

# Clean up build dependencies to reduce image size
RUN apk del build-base gcc g++ gfortran cmake && \
    rm -rf /var/cache/apk/* && \
    rm -rf /root/.cache/pip/*

# Set working directory
WORKDIR /app

# Copy your application code
COPY . .

# Default command
CMD ["python"]