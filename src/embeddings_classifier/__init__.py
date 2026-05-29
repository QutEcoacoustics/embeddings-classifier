"""Embeddings classifier package."""

from importlib.metadata import PackageNotFoundError, version
from .app import (
    ClassifierConfig,
    ClassifierResult,
    classify,
    classify_dataframe,
    classify_table,
    process_single_file,
)

try:
    __version__ = version("embeddings-classifier")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "ClassifierConfig",
    "ClassifierResult",
    "classify",
    "classify_table",
    "classify_dataframe",
    "process_single_file",
]
