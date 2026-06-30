
from collections import Counter
import json
import base64
import binascii
import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Iterator
from dataclasses import dataclass
import re

import numpy as np




def resolve_classifier_name(config: Dict[str, Any], index: int, fail_on_missing: bool = False) -> str:
    """Resolve a classifier name from any one of a few possible keys. 
       If it's not set, generate a name based on the classes or index.
    """
    configured_name = config.get('classifier_name') or config.get('name')

    if isinstance(configured_name, str) and configured_name.strip():
        return configured_name.strip()
    elif fail_on_missing:
        raise ValueError("Unable to resolve classifier name from config.")

    classes = config.get('classifier', {}).get('classes', [])
    if isinstance(classes, list) and len(classes) == 1:
        class_name = re.sub(r'\s+', '_', str(classes[0]).strip().lower())
        return class_name if class_name else f"classifier_{index}"

    if isinstance(classes, list) and len(classes) > 1:
        return f"classifier_{index}_{len(classes)}class"

    return f"classifier_{index}"


def deserialize_classifier_params(classifier_config: Dict[str, Any]) -> tuple:
    """Deserialize beta weights and bias from base64 encoded strings."""

    def do_decode(x: str) -> np.ndarray:
        flat = np.frombuffer(base64.b64decode(x.encode('ascii')), dtype=np.float32)
        return flat

    try:
        beta_flat = do_decode(classifier_config['beta'])
        num_rows = len(beta_flat) // len(classifier_config['classes'])
        num_cols = len(classifier_config['classes'])
        beta = beta_flat.reshape(num_rows, num_cols)
        beta_bias = do_decode(classifier_config['beta_bias'])
        return beta, beta_bias

    except (ValueError, TypeError, KeyError, binascii.Error) as e:
        logging.error("Error deserializing classifier parameters: %s", e)
        raise ValueError(f"Error deserializing classifier parameters: {e}") from e


def build_threshold_array(classes: List[str], threshold: Any) -> np.ndarray:
    """Return one float32 threshold per class.

    Accepted threshold formats:
    - scalar number: applied to all classes
    - None: no threshold for all classes
    - list: one value per class (number or None)
    - dict: class -> value (number or None); missing classes default to 0.0
    """

    default_threshold = 0.0

    # Canonicalize thresholds so we always have one value per class.
    # Allowed per-class values are float (enforce threshold) or None (no threshold for that class).
    if isinstance(threshold, dict):
        # Missing classes default to 0.0 for consistency with existing behavior.
        per_class_thresholds = {cls: threshold.get(cls, default_threshold) for cls in classes}
    elif isinstance(threshold, list):
        if len(threshold) != len(classes):
            raise ValueError(
                f"Threshold list length ({len(threshold)}) must match number of classes ({len(classes)})"
            )
        per_class_thresholds = dict(zip(classes, threshold))
    elif threshold is None or isinstance(threshold, (int, float, np.integer, np.floating)):
        per_class_thresholds = {cls: threshold for cls in classes}
    else:
        raise TypeError(
            "Threshold must be one of: float/int, None, list (one per class), "
            "or dict mapping class name to threshold."
        )

    for cls, value in per_class_thresholds.items():
        if value is None:
            continue
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"Threshold for class '{cls}' must be a number or None, got {type(value).__name__}."
            )
        per_class_thresholds[cls] = float(value)

    # None means no threshold for that class, so use the minimum float32 sentinel.
    return np.array(
        [
            np.finfo(np.float32).min if per_class_thresholds[cls] is None else per_class_thresholds[cls]
            for cls in classes
        ],
        dtype=np.float32,
    )


@dataclass
class ClassifierConfig:
    """Normalized configuration for a single classifier."""

    # classifier_properties

    classifier_name: str
    classes: List[str]
    beta: np.ndarray
    beta_bias: np.ndarray
    model_config: Dict[str, Any] = None  # Optional additional model config

    # run params
    threshold: Any = 0.0
    threshold_array: np.ndarray = None
    save_empty: bool = True
    skip_existing: bool = True

    def __post_init__(self) -> None:
        # Keep classes as a concrete list for deterministic ordering and serialization.
        self.classes = list(self.classes)

        # Support both encoded (str) and already-materialized (ndarray/list) params.
        if isinstance(self.beta, str) or isinstance(self.beta_bias, str):
            if not (isinstance(self.beta, str) and isinstance(self.beta_bias, str)):
                raise TypeError("beta and beta_bias must both be encoded strings or both be array-like")
            self.beta, self.beta_bias = deserialize_classifier_params(
                {
                    'beta': self.beta,
                    'beta_bias': self.beta_bias,
                    'classes': self.classes,
                }
            )
        else:
            self.beta = np.asarray(self.beta, dtype=np.float32)
            self.beta_bias = np.asarray(self.beta_bias, dtype=np.float32)

        if self.beta.ndim != 2:
            raise ValueError(f"beta must be a 2D array, got shape {self.beta.shape}")

        if self.beta.shape[1] != len(self.classes):
            raise ValueError(
                "beta shape does not match classes: "
                f"expected {len(self.classes)} columns, got {self.beta.shape[1]}"
            )

        # Accept common bias layouts and canonicalize to 1D [num_classes].
        self.beta_bias = np.asarray(self.beta_bias, dtype=np.float32).reshape(-1)
        if self.beta_bias.shape[0] != len(self.classes):
            raise ValueError(
                "beta_bias shape does not match classes: "
                f"expected {len(self.classes)} values, got {self.beta_bias.shape[0]}"
            )

        # Canonical per-class threshold array is computed once at normalization.
        if self.threshold_array is None:
            self.threshold_array = build_threshold_array(self.classes, self.threshold)
        else:
            self.threshold_array = np.asarray(self.threshold_array, dtype=np.float32)

        if self.threshold_array.shape != (len(self.classes),):
            raise ValueError(
                "threshold_array shape does not match classes: "
                f"expected {(len(self.classes),)}, got {self.threshold_array.shape}"
            )

        self.classifier_name = str(self.classifier_name).strip()
        if not self.classifier_name:
            raise ValueError("classifier_name cannot be empty")
        self.save_empty = bool(self.save_empty)
        self.skip_existing = bool(self.skip_existing)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], index: int = 0) -> "ClassifierConfig":
        if not isinstance(config, dict):
            raise ValueError(f"Config at index {index} must be an object")

        # Validate required fields
        if 'classifier' not in config:
            # Assume the entire JSON is the classifier and use defaults.
            if 'classes' not in config:
                raise ValueError("Config must contain 'classifier' section")
            config = {
                'classifier': config
            }

        classifier = config['classifier']
        required_fields = ['classes', 'beta', 'beta_bias']
        for field in required_fields:
            if field not in classifier:
                raise ValueError(f"Classifier must contain '{field}' field")

        # Canonical classifier name is resolved once during normalization.
        classifier_name = resolve_classifier_name(config, index)

        normalized = cls(
            classifier_name=classifier_name,
            classes=classifier['classes'],
            beta=classifier['beta'],
            beta_bias=classifier['beta_bias'],
            model_config=config.get('model_config'),
            threshold=config.get('threshold', 0.0),
            threshold_array=config.get('threshold_array'),
            save_empty=config.get('save_empty', True),
            skip_existing=config.get('skip_existing', True),
        )

        logging.info("Classes found in config: %s", normalized.classes)
        logging.info("Beta shape: %s, Beta bias shape: %s", normalized.beta.shape, normalized.beta_bias.shape)

        return normalized

    @classmethod
    def from_json(cls, config_path: Union[str, Path], index: int = 0) -> "ClassifierConfig":
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            raise ValueError("Expected single config object in JSON, got a list. Use ClassifierConfigList.")
        return cls.from_dict(raw, index=index)

    @classmethod
    def from_any(
        cls,
        config_data: Union["ClassifierConfig", Dict[str, Any], str, Path],
        index: int = 0,
    ) -> "ClassifierConfig":
        if isinstance(config_data, cls):
            return config_data

        if isinstance(config_data, (str, Path)):
            return cls.from_json(config_data, index=index)

        if not isinstance(config_data, dict):
            raise TypeError(f"Unsupported config type: {type(config_data)}")

        return cls.from_dict(config_data, index=index)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'classifier': {
                'classes': self.classes,
                'beta': self.beta,
                'beta_bias': self.beta_bias,
            },
            'classifier_name': self.classifier_name,
            'threshold_array': self.threshold_array,
            'save_empty': self.save_empty,
            'skip_existing': self.skip_existing,
        }


@dataclass
class ClassifierConfigList:
    """Collection of normalized classifier configs with list-level validation helpers."""

    configs: List[ClassifierConfig]

    def __post_init__(self) -> None:
        self.ensure_unique_classifier_names()

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> "ClassifierConfigList":
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return cls.from_any(raw)

    @classmethod
    def from_any(
        cls,
        config_data: Union[
            "ClassifierConfigList",
            ClassifierConfig,
            Dict[str, Any],
            List[Any],
            str,
            Path,
        ],
    ) -> "ClassifierConfigList":
        if isinstance(config_data, cls):
            return config_data

        if isinstance(config_data, (str, Path)):
            return cls.from_json(config_data)

        if isinstance(config_data, ClassifierConfig):
            return cls(configs=[config_data])

        if isinstance(config_data, dict):
            return cls(configs=[ClassifierConfig.from_dict(config_data, index=0)])

        if isinstance(config_data, list):
            normalized = []
            for i, config in enumerate(config_data):
                if isinstance(config, ClassifierConfig):
                    normalized.append(config)
                else:
                    normalized.append(ClassifierConfig.from_dict(config, index=i))
            return cls(configs=normalized)

        raise TypeError(f"Unsupported config type: {type(config_data)}")

    def ensure_unique_classifier_names(self) -> None:
        names = [config.classifier_name for config in self.configs]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            duplicates_str = ', '.join(duplicates)
            raise ValueError(f"Duplicate classifier_name values are not allowed: {duplicates_str}")

    def as_list(self) -> List[Dict[str, Any]]:
        return [config.as_dict() for config in self.configs]

    def __iter__(self) -> Iterator[ClassifierConfig]:
        return iter(self.configs)

    def __len__(self) -> int:
        return len(self.configs)
