"""Embeddings classifier package."""

from importlib.metadata import PackageNotFoundError, version
from .config import (
    ClassifierConfig,
    ClassifierConfigList,
)
from .app import (
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
    "ClassifierConfigList",
    "ClassifierItem",
    "classify",
    "classify_table",
    "classify_dataframe"
]
