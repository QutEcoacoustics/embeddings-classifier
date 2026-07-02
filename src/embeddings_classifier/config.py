
from collections import Counter
import json
import base64
import binascii
import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Iterator, Optional
from dataclasses import dataclass
import re

import numpy as np




RUN_PARAM_KEYS = {'threshold', 'threshold_array', 'save_empty', 'skip_existing'}


def _extract_run_params_from_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract run params from both top-level keys and optional nested run_config."""
    result: Dict[str, Any] = {}
    for key in RUN_PARAM_KEYS:
        if key in data:
            result[key] = data[key]

    nested_run = data.get('run_config')
    if isinstance(nested_run, dict):
        for key in RUN_PARAM_KEYS:
            if key in nested_run:
                result[key] = nested_run[key]

    return result


def normalize_single_config_schema(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single config object to canonical schema.

    Canonical output shape:
    {
      'classifier': {
        'classes', 'beta', 'beta_bias', 'classifier_name', 'model_config'
      },
      'run_config': {
        'threshold', 'threshold_array', 'save_empty', 'skip_existing'
      }
    }
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")

    def from_options(options: List[str], data) -> Optional[Any]:
        # for a list of possible keys, return the value of the first matching one
        # reject configs that specify more than one alias with different values.
        found_key = None
        found_value = None
        for key in options:
            if key in data:
                value = data[key]
                if found_key is None:
                    found_key = key
                    found_value = value
                    continue

                if value != found_value:
                    raise ValueError(f"Multiple keys found: {found_key}, {key}")
        return found_value
    
    def check_nested(key, nested_dict_name, names=None):
        if names is None:
            names = [key]
        # checks the config for the key in either the nested dict or the top-level config
        nested_dict = config.get(nested_dict_name)
        if isinstance(nested_dict, dict):
            if any(key in nested_dict for key in names):
                return from_options(names, nested_dict)
        return from_options(names, config)
    
    classifier_name_key_options = ['classifier_name', 'name', 'recognizer_name']

    classifier_names = ['classes', 'beta', 'beta_bias', 'model_config']
    run_names = ['threshold', 'threshold_array', 'save_empty', 'skip_existing']

    classifier = {key: check_nested(key, 'classifier') for key in classifier_names}
    classifier['classifier_name'] = check_nested('classifier_name', 'classifier', classifier_name_key_options)
    
    run_config = {key: check_nested(key, 'run_config') for key in run_names}
    # Only include run_config keys that have non-None values
    run_config = {key: value for key, value in run_config.items() if value is not None}

    return {
        'classifier': classifier,
        'run_config': run_config,
    }


def _validate_container_run_config(container_run_config: Optional[Any], recognizer_count: int) -> None:
    """Validate outer run_config shape before classifier-level construction."""
    if container_run_config is None:
        return

    if isinstance(container_run_config, list):
        if len(container_run_config) != recognizer_count:
            raise ValueError(
                "Container run_config list length must match recognizer count: "
                f"{len(container_run_config)} != {recognizer_count}"
            )
        return

    if isinstance(container_run_config, dict):
        # can be either a flat run-param dict or a name-keyed dict for per-classifier overrides, but not both
        has_run_param_keys = any(key in RUN_PARAM_KEYS for key in container_run_config.keys())
        has_non_run_param_keys = any(key not in RUN_PARAM_KEYS for key in container_run_config.keys())
        if has_run_param_keys and has_non_run_param_keys:
            raise ValueError(
                "Container run_config cannot mix run-param keys with classifier-name keys"
            )
        
        # If name-keyed (has non-run-param keys), validate all values are dicts or None
        if has_non_run_param_keys:
            for name, value in container_run_config.items():
                if value is not None and not isinstance(value, dict):
                    raise ValueError(
                        f"Container run_config entry for classifier '{name}' must be an object or None"
                    )
        return

    raise ValueError("Container run_config must be an object or list")


def _resolve_global_run_config_for_classifier(
    global_run_config: Optional[Any],
    index: int,
    classifier_name: str,
) -> Optional[Dict[str, Any]]:
    """Resolve applicable global run-config override for one classifier."""
    if global_run_config is None:
        return None

    if isinstance(global_run_config, list):
        selected = global_run_config[index]
        if selected is None:
            return None
        if not isinstance(selected, dict):
            raise ValueError("Each run_config list entry must be an object")
        return _extract_run_params_from_mapping(selected)

    if isinstance(global_run_config, dict):
        # Flat run-param dict applies to all classifiers.
        if any(key in RUN_PARAM_KEYS for key in global_run_config.keys()):
            return _extract_run_params_from_mapping(global_run_config)

        # Name-keyed dict: map by resolved classifier_name.
        if classifier_name not in global_run_config:
            return None

        selected = global_run_config[classifier_name]
        if selected is None:
            return None
        if not isinstance(selected, dict):
            raise ValueError(
                f"Container run_config entry for classifier '{classifier_name}' must be an object or None"
            )
        return _extract_run_params_from_mapping(selected)

    raise ValueError("Container run_config must be an object or list")


def resolve_classifier_name(config: Dict[str, Any], index: int, fail_on_missing: bool = False) -> str:
    """Resolve classifier name from canonical schema.
       If the explicit 
    ``classifier_name`` is missing, fallback naming is derived from classes/index.
    """
    classifier = config['classifier']

    configured_name = classifier.get('classifier_name')

    if isinstance(configured_name, str) and configured_name.strip():
        return configured_name.strip()
    elif fail_on_missing:
        raise ValueError("Unable to resolve classifier name from config.")

    classes = classifier.get('classes', [])
    if isinstance(classes, list) and len(classes) == 1:
        class_name = re.sub(r'\s+', '_', str(classes[0]).strip().lower())
        return class_name if class_name else f"classifier_{index}"

    if isinstance(classes, list) and len(classes) > 1:
        return f"classifier_{index}_{len(classes)}class"

    return f"classifier_{index}"


def deserialize_classifier_params(classifier_config: Dict[str, Any]) -> tuple:
    """Deserialize beta weights and bias from base64 encoded strings."""

    def do_decode(x: str) -> np.ndarray:
        raw = base64.b64decode(x.strip().encode('ascii'), validate=True)
        return np.frombuffer(raw, dtype=np.float32)

    try:
        beta_flat = do_decode(classifier_config['beta'])
        num_cols = len(classifier_config['classes'])
        if num_cols == 0:
            raise ValueError("Classifier must contain at least one class")

        if len(beta_flat) % num_cols != 0:
            raise ValueError(
                "Invalid beta length: "
                f"{len(beta_flat)} values is not divisible by {num_cols} classes"
            )

        num_rows = len(beta_flat) // num_cols
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


@dataclass(frozen=True)
class ClassifierConfig:
    """Normalized configuration for a single classifier."""

    # classifier_properties

    classifier_name: str
    classes: List[str]
    beta: np.ndarray
    beta_bias: np.ndarray
    model_config: Optional[Dict[str, Any]] = None  # Optional additional model config

    # run params
    threshold: Any = 0.0
    threshold_array: Optional[np.ndarray] = None
    save_empty: bool = True
    skip_existing: bool = True

    def __post_init__(self) -> None:
        # Keep classes as a concrete list for deterministic ordering and serialization.
        object.__setattr__(self, 'classes', list(self.classes))

        # Support both encoded (str) and already-materialized (ndarray/list) params.
        if isinstance(self.beta, str) or isinstance(self.beta_bias, str):
            if not (isinstance(self.beta, str) and isinstance(self.beta_bias, str)):
                raise TypeError("beta and beta_bias must both be encoded strings or both be array-like")
            beta, beta_bias = deserialize_classifier_params(
                {
                    'beta': self.beta,
                    'beta_bias': self.beta_bias,
                    'classes': self.classes,
                }
            )
            object.__setattr__(self, 'beta', beta)
            object.__setattr__(self, 'beta_bias', beta_bias)
        else:
            object.__setattr__(self, 'beta', np.asarray(self.beta, dtype=np.float32))
            object.__setattr__(self, 'beta_bias', np.asarray(self.beta_bias, dtype=np.float32))

        if self.beta.ndim != 2:
            raise ValueError(f"beta must be a 2D array, got shape {self.beta.shape}")

        if self.beta.shape[1] != len(self.classes):
            raise ValueError(
                "beta shape does not match classes: "
                f"expected {len(self.classes)} columns, got {self.beta.shape[1]}"
            )

        # Accept common bias layouts and canonicalize to 1D [num_classes].
        object.__setattr__(self, 'beta_bias', np.asarray(self.beta_bias, dtype=np.float32).reshape(-1))
        if self.beta_bias.shape[0] != len(self.classes):
            raise ValueError(
                "beta_bias shape does not match classes: "
                f"expected {len(self.classes)} values, got {self.beta_bias.shape[0]}"
            )

        # Canonical per-class threshold array is computed once at normalization.
        if self.threshold_array is None:
            object.__setattr__(self, 'threshold_array', build_threshold_array(self.classes, self.threshold))
        else:
            object.__setattr__(self, 'threshold_array', np.asarray(self.threshold_array, dtype=np.float32))

        # At this point, threshold_array is guaranteed to be an ndarray (never None)
        assert isinstance(self.threshold_array, np.ndarray)
        if self.threshold_array.shape != (len(self.classes),):
            raise ValueError(
                "threshold_array shape does not match classes: "
                f"expected {(len(self.classes),)}, got {self.threshold_array.shape}"
            )

        object.__setattr__(self, 'classifier_name', str(self.classifier_name).strip())
        if not self.classifier_name:
            raise ValueError("classifier_name cannot be empty")
        object.__setattr__(self, 'save_empty', bool(self.save_empty))
        object.__setattr__(self, 'skip_existing', bool(self.skip_existing))

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        index: int = 0,
        run_config: Optional[Any] = None,
    ) -> "ClassifierConfig":
        if not isinstance(config, dict):
            raise ValueError(f"Config at index {index} must be an object")

        normalized_input = normalize_single_config_schema(config)
        classifier = normalized_input['classifier']
        required_fields = ['classes', 'beta', 'beta_bias']
        for field in required_fields:
            if field not in classifier:
                raise ValueError(f"Classifier must contain '{field}' field")
            if classifier[field] is None:
                raise ValueError(f"Classifier field '{field}' cannot be null")

        # Canonical classifier name is resolved once during normalization.
        classifier_name = resolve_classifier_name(normalized_input, index)

        # Start with per-recognizer run_config, apply global overrides
        merged_run_config = dict(normalized_input.get('run_config', {}))
        global_override = _resolve_global_run_config_for_classifier(
            run_config,
            index=index,
            classifier_name=classifier_name,
        )
        if isinstance(global_override, dict):
            merged_run_config.update(global_override)

        normalized = cls(
            classifier_name=classifier_name,
            classes=classifier['classes'],
            beta=classifier['beta'],
            beta_bias=classifier['beta_bias'],
            model_config=classifier.get('model_config'),
            threshold=merged_run_config.get('threshold', 0.0),
            threshold_array=merged_run_config.get('threshold_array'),
            save_empty=merged_run_config.get('save_empty', True),
            skip_existing=merged_run_config.get('skip_existing', True),
        )

        logging.info("Classes found in config: %s", normalized.classes)
        logging.info("Beta shape: %s, Beta bias shape: %s", normalized.beta.shape, normalized.beta_bias.shape)

        return normalized

    @classmethod
    def from_json(
        cls,
        config_path: Union[str, Path],
        index: int = 0,
        run_config: Optional[Any] = None,
    ) -> "ClassifierConfig":
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            raise ValueError("Expected single config object in JSON, got a list. Use ClassifierConfigList.")
        return cls.from_dict(raw, index=index, run_config=run_config)

    @classmethod
    def from_any(
        cls,
        config_data: Union["ClassifierConfig", Dict[str, Any], str, Path],
        index: int = 0,
        run_config: Optional[Any] = None,
    ) -> "ClassifierConfig":
        if isinstance(config_data, cls):
            return config_data

        if isinstance(config_data, (str, Path)):
            return cls.from_json(config_data, index=index, run_config=run_config)

        if not isinstance(config_data, dict):
            raise TypeError(f"Unsupported config type: {type(config_data)}")

        return cls.from_dict(config_data, index=index, run_config=run_config)

    @property
    def embedding_model_name(self) -> Optional[str]:
        model_config = self.model_config if isinstance(self.model_config, dict) else {}

        # perch model config schema
        perch_model_version = model_config.get('model_config', {}).get('tfhub_version')
        if str(perch_model_version) in ["4", "8"]:
            return "perch_8"

        # infer model name from embedding dimensions
        logging.warning("Embedding model name not explicitly set in config. Inferring from beta shape.")
        shape_map = {
            1280: "perch_8",
            1536: "perch_v2",
            1024: "birdnet_v2.4"

        }
        return shape_map.get(self.beta.shape[0], None)
        

    def as_dict(self) -> Dict[str, Any]:
        return {
            'classifier': {
                'classes': self.classes,
                'beta': self.beta,
                'beta_bias': self.beta_bias,
                'classifier_name': self.classifier_name,
                'model_config': self.model_config,
            },
            "run_config": {
                'threshold': self.threshold,
                'threshold_array': self.threshold_array,
                'save_empty': self.save_empty,
                'skip_existing': self.skip_existing,
            }
        }


@dataclass
class ClassifierConfigList:
    """Collection of normalized classifier configs with list-level validation helpers."""

    configs: List[ClassifierConfig]

    def __post_init__(self) -> None:
        self.ensure_unique_classifier_names()
        self.ensure_compatible_embedding_model_names()

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
            if 'recognizers' in config_data:
                recognizers = config_data['recognizers']
                if isinstance(recognizers, dict):
                    recognizer_configs = [recognizers]
                elif isinstance(recognizers, list):
                    recognizer_configs = recognizers
                else:
                    raise ValueError("'recognizers' must be an object or list")

                container_run_config = config_data.get('run_config')
                if container_run_config is None:
                    # Allow flat run params at wrapper level too.
                    flat_run = _extract_run_params_from_mapping(config_data)
                    container_run_config = flat_run if flat_run else None

                _validate_container_run_config(
                    container_run_config,
                    recognizer_count=len(recognizer_configs),
                )

                normalized = []
                for i, recognizer in enumerate(recognizer_configs):
                    normalized.append(
                        ClassifierConfig.from_dict(
                            recognizer,
                            index=i,
                            run_config=container_run_config,
                        )
                    )
                return cls(configs=normalized)

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

    def ensure_compatible_embedding_model_names(self) -> None:
        """accesses a property that raises if there are multiple different
          embedding_model_names across configs."""
        _ = self.embedding_model_name


    @property
    def embedding_model_name(self) -> Optional[str]:
        embedding_model_names = {
            model_name for model_name in (config.embedding_model_name for config in self.configs)
            if model_name is not None
        }
        if not embedding_model_names:
            return None
        elif len(embedding_model_names) > 1:
            raise ValueError(
                "All classifier configs must have the same embedding_model_name or None: "
                f"{', '.join(sorted(embedding_model_names))}"
            )
        return next(iter(embedding_model_names))

    def as_list(self) -> List[Dict[str, Any]]:
        return [config.as_dict() for config in self.configs]

    def __iter__(self) -> Iterator[ClassifierConfig]:
        return iter(self.configs)

    def __len__(self) -> int:
        return len(self.configs)
