"""Embeddings classifier package."""

from importlib.metadata import PackageNotFoundError, version
from .app import (
    ClassifierConfig,
    ClassifierItem,
    classify,
    classify_dataframe,
    classify_table
)

try:
    __version__ = version("embeddings-classifier")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "ClassifierConfig",
    "ClassifierItem",
    "classify",
    "classify_table",
    "classify_dataframe"
]
