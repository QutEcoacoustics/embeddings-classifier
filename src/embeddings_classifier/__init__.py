"""Embeddings classifier package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("embeddings-classifier")
except PackageNotFoundError:
    __version__ = "0.0.0"
