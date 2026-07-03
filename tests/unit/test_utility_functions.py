
import json
import copy
import dataclasses
import base64
from pathlib import Path
import pytest

import numpy as np


import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import TestHelpers
from unit_helpers import UnitTestHelpers
import embeddings_classifier.app as app
import embeddings_classifier.config as config_module

class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_load_config_valid(self, sample_data):
        """Test loading valid configuration."""
        config_list = app.load_config(sample_data['config_path'])

        assert isinstance(config_list, list)

        for config in config_list:
            assert 'classifier' in config
            assert 'classes' in config['classifier']
            assert 'beta' in config['classifier']
            assert 'beta_bias' in config['classifier']
    
    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON configuration."""
        config_path = tmp_path / "invalid_config.json"
        config_path.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            app.load_config(str(config_path))
    

    def test_load_config_missing_classifier(self, tmp_path):
        """Test loading config missing classifier section."""
        config_path = tmp_path / "no_classifier.json"
        config_path.write_text('{"other": "data"}')
        
        with pytest.raises(ValueError):
            app.load_config(str(config_path))
    

    def test_load_config_missing_required_fields(self, tmp_path):
        """Test loading config missing required fields."""
        config_path = tmp_path / "incomplete_config.json"
        config_data = {
            "classifier": {
                "classes": ["test"],
                "beta": "some_data"
                # Missing beta_bias
            }
        }
        config_path.write_text(json.dumps(config_data))
        
        with pytest.raises(ValueError):
            app.load_config(str(config_path))
    

    def test_deserialize_classifier_params(self, sample_data):
        """Test deserializing classifier parameters."""
        classifier = sample_data['config']['classifier']
        beta, beta_bias = config_module.deserialize_classifier_params(classifier)
        
        assert isinstance(beta, np.ndarray)
        assert isinstance(beta_bias, np.ndarray)
        assert beta.dtype == np.float32
        assert beta_bias.dtype == np.float32
    

    def test_deserialize_classifier_params_invalid_base64(self, tmp_path):
        """Test deserializing invalid base64 data."""
        classifier = {
            "classes": ["test"],
            "beta": "invalid_base64!@#",
            "beta_bias": "also_invalid!@#"
        }

        with pytest.raises(ValueError):
            config_module.deserialize_classifier_params(classifier)

    def test_deserialize_classifier_params_non_divisible_beta_length(self):
        """beta length must be divisible by class count for matrix reshape."""
        beta_flat = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # 5 values
        beta_bias = np.array([0, 0], dtype=np.float32)
        classifier = {
            "classes": ["a", "b"],  # 2 classes, so 5 is invalid
            "beta": base64.b64encode(beta_flat.tobytes()).decode('ascii'),
            "beta_bias": base64.b64encode(beta_bias.tobytes()).decode('ascii'),
        }

        with pytest.raises(ValueError, match="not divisible"):
            config_module.deserialize_classifier_params(classifier)
    

    def test_get_parquet_files(self, clean_mounted_dirs):
        """Test getting parquet files from directory."""
        dirs = clean_mounted_dirs
        input_dir = dirs['workspace_input']
        
        # Create some parquet files
        UnitTestHelpers.create_sample_parquet(input_dir / 'file1.parquet')
        UnitTestHelpers.create_sample_parquet(input_dir / 'subdir' / 'file2.parquet')
        
        # Create a non-parquet file
        (input_dir / 'readme.txt').write_text('test')
        
        parquet_files = app.get_parquet_files(str(input_dir))
        
        assert len(parquet_files) == 2
        assert any('file1.parquet' in f.name for f in parquet_files)
        assert any('file2.parquet' in f.name for f in parquet_files)
        assert not any('readme.txt' in f.name for f in parquet_files)

    def test_init_items_raises_on_sanitized_name_collision(self, sample_data):
        """Colliding sanitized classifier names should fail fast with a clear error."""
        first = list(config_module.ClassifierConfigList.from_any(sample_data['config_path']))[0]
        first = dataclasses.replace(first, classifier_name='A B')
        second = dataclasses.replace(first, classifier_name='A_B')
        configs = config_module.ClassifierConfigList(configs=[first, second])

        with pytest.raises(ValueError, match='resolve to the same output path'):
            app.init_items(configs, Path(sample_data['output_dir']) / '<classifier_name>' / 'result.csv')

    def test_resolve_classifier_name_raises_when_missing_and_required(self):
        """Missing explicit classifier_name should raise when fail_on_missing is True."""
        config = {
            'classifier': {'classes': ['owl']},
        }
        with pytest.raises(ValueError, match='Unable to resolve classifier name from config'):
            config_module.resolve_classifier_name(config, 0, fail_on_missing=True)

    def test_resolve_classifier_name_single_class_fallback(self):
        """Single-class classifiers should derive name from class label."""
        config = {
            'classifier': {'classes': ['Yellow bellied glider']},
        }
        assert config_module.resolve_classifier_name(config, 2) == 'yellow_bellied_glider'

    def test_resolve_classifier_name_multi_class_fallback(self):
        """Multi-class classifiers should use informative short fallback names."""
        config = {
            'classifier': {'classes': ['a', 'b', 'c']},
        }
        assert config_module.resolve_classifier_name(config, 1) == 'classifier_1_3class'

    def test_init_items_raises_on_derived_name_collision(self, sample_data):
        """Collisions from class-derived fallback names should be detected."""
        base = copy.deepcopy(sample_data['config'])
        base.pop('classifier_name', None)
        base.pop('name', None)
        base['classifier'] = dict(base['classifier'])
        base['classifier']['classes'] = ['A B']

        second = copy.deepcopy(base)
        second['classifier']['classes'] = ['a_b']

        with pytest.raises(ValueError, match='Duplicate classifier_name values are not allowed'):
            config_module.ClassifierConfigList.from_any([base, second])

    def test_build_threshold_array_scalar(self):
        classes = ['a', 'b']
        result = config_module.build_threshold_array(classes, 0.25)
        np.testing.assert_allclose(result, np.array([0.25, 0.25], dtype=np.float32))

    def test_build_threshold_array_none(self):
        classes = ['a', 'b']
        result = config_module.build_threshold_array(classes, None)
        expected = np.array([np.finfo(np.float32).min, np.finfo(np.float32).min], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_build_threshold_array_list_and_length_validation(self):
        classes = ['a', 'b']
        result = config_module.build_threshold_array(classes, [0.1, None])
        expected = np.array([0.1, np.finfo(np.float32).min], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

        with pytest.raises(ValueError, match='Threshold list length'):
            config_module.build_threshold_array(classes, [0.1])

    def test_build_threshold_array_dict_with_missing_and_none(self):
        classes = ['a', 'b', 'c']
        result = config_module.build_threshold_array(classes, {'a': 0.2, 'b': None})
        expected = np.array([0.2, np.finfo(np.float32).min, 0.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_build_threshold_array_invalid_types(self):
        classes = ['a']

        with pytest.raises(TypeError, match='Threshold must be one of'):
            config_module.build_threshold_array(classes, 'bad')

        with pytest.raises(TypeError, match="Threshold for class 'a' must be a number or None"):
            config_module.build_threshold_array(classes, {'a': 'bad'})

    def test_from_any_does_not_mutate_input_with_threshold_array(self, sample_data):
        raw = copy.deepcopy(sample_data['config'])

        assert 'threshold_array' not in raw

        config = config_module.ClassifierConfig.from_any(raw)

        assert isinstance(config.threshold_array, np.ndarray)
        assert 'threshold_array' not in raw

    def test_as_dict_preserves_model_config_and_threshold(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a', 'b'],
            beta=np.zeros((2, 2), dtype=np.float32),
            beta_bias=np.zeros(2, dtype=np.float32),
            model_config={'foo': 'bar'},
            threshold={'a': 0.1, 'b': 0.2},
        )

        serialized = config.as_dict()

        assert serialized['classifier']['model_config'] == {'foo': 'bar'}
        assert serialized['run_config']['threshold'] == {'a': 0.1, 'b': 0.2}

    def test_classifier_config_is_frozen(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.classifier_name = 'new_name'

    def test_resolve_classifier_name_uses_nested_classifier_name(self):
        config = {
            'classifier': {
                'classes': ['owl'],
                'classifier_name': 'Nested Owl Name',
            }
        }
        assert config_module.resolve_classifier_name(config, 0) == 'Nested Owl Name'

    def test_classifier_config_list_from_recognizers_with_flat_global_run_config(self):
        beta = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')
        beta_bias = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')

        payload = {
            'recognizers': [
                {
                    'classifier': {
                        'classes': ['a'],
                        'beta': beta,
                        'beta_bias': beta_bias,
                    }
                }
            ],
            'run_config': {
                'save_empty': False,
                'skip_existing': False,
                'threshold': 0.3,
            },
        }

        configs = config_module.ClassifierConfigList.from_any(payload)
        cfg = list(configs)[0]
        assert cfg.save_empty is False
        assert cfg.skip_existing is False
        np.testing.assert_allclose(cfg.threshold_array, np.array([0.3], dtype=np.float32))

    def test_classifier_config_list_from_recognizers_with_name_keyed_run_config(self):
        beta = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')
        beta_bias = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')

        payload = {
            'recognizers': [
                {
                    'classifier': {
                        'classes': ['a'],
                        'beta': beta,
                        'beta_bias': beta_bias,
                        'classifier_name': 'alpha',
                    }
                },
                {
                    'classifier': {
                        'classes': ['b'],
                        'beta': beta,
                        'beta_bias': beta_bias,
                        'classifier_name': 'beta',
                    }
                },
            ],
            'run_config': {
                'beta': {'threshold': 0.8},
            },
        }

        configs = list(config_module.ClassifierConfigList.from_any(payload))
        np.testing.assert_allclose(configs[0].threshold_array, np.array([0.0], dtype=np.float32))
        np.testing.assert_allclose(configs[1].threshold_array, np.array([0.8], dtype=np.float32))

    def test_classifier_config_list_from_recognizers_with_parallel_run_config(self):
        beta = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')
        beta_bias = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')

        payload = {
            'recognizers': [
                {'classifier': {'classes': ['a'], 'beta': beta, 'beta_bias': beta_bias}},
                {'classifier': {'classes': ['b'], 'beta': beta, 'beta_bias': beta_bias}},
            ],
            'run_config': [
                {'threshold': 0.1},
                {'threshold': 0.9},
            ],
        }

        configs = list(config_module.ClassifierConfigList.from_any(payload))
        np.testing.assert_allclose(configs[0].threshold_array, np.array([0.1], dtype=np.float32))
        np.testing.assert_allclose(configs[1].threshold_array, np.array([0.9], dtype=np.float32))

    def test_classifier_config_list_rejects_mixed_global_run_config_dict(self):
        beta = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')
        beta_bias = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')

        payload = {
            'recognizers': [
                {'classifier': {'classes': ['a'], 'beta': beta, 'beta_bias': beta_bias}},
            ],
            'run_config': {
                'threshold': 0.2,
                'koala': {'threshold': 0.9},
            },
        }

        with pytest.raises(ValueError, match='cannot mix run-param keys with classifier-name keys'):
            config_module.ClassifierConfigList.from_any(payload)

    def test_classifier_config_list_rejects_non_dict_name_keyed_value(self):
        beta = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')
        beta_bias = base64.b64encode(np.array([0.0], dtype=np.float32).tobytes()).decode('ascii')

        payload = {
            'recognizers': [
                {
                    'classifier': {
                        'classes': ['a'],
                        'beta': beta,
                        'beta_bias': beta_bias,
                        'classifier_name': 'koala',
                    }
                }
            ],
            'run_config': {
                'koala': 0.5,
            },
        }

        with pytest.raises(ValueError, match="must be an object or None"):
            config_module.ClassifierConfigList.from_any(payload)

    def test_classifier_config_rejects_beta_columns_mismatch(self):
        with pytest.raises(ValueError, match='beta shape does not match classes'):
            config_module.ClassifierConfig(
                classifier_name='bad_beta',
                classes=['a', 'b'],
                beta=np.zeros((3, 1), dtype=np.float32),
                beta_bias=np.zeros(2, dtype=np.float32),
            )

    def test_classifier_config_rejects_threshold_array_length_mismatch(self):
        with pytest.raises(ValueError, match='threshold_array shape does not match classes'):
            config_module.ClassifierConfig(
                classifier_name='bad_threshold',
                classes=['a', 'b'],
                beta=np.zeros((3, 2), dtype=np.float32),
                beta_bias=np.zeros(2, dtype=np.float32),
                threshold_array=np.array([0.1], dtype=np.float32),
            )

    def test_classifier_config_embedding_model_name_defaults_to_none(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        assert config.embedding_model_name is None

    def test_classifier_config_sets_embedding_dim_tuple(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a'],
            beta=np.zeros((1536, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        assert config.embedding_dim == (1536,)

    def test_classifier_config_list_embedding_model_name_returns_none_when_all_none(self):
        first = config_module.ClassifierConfig(
            classifier_name='first',
            classes=['a'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )
        second = config_module.ClassifierConfig(
            classifier_name='second',
            classes=['b'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        configs = config_module.ClassifierConfigList(configs=[first, second])

        assert configs.embedding_model_name is None

    def test_classifier_config_list_embedding_dim_returns_unique_value(self):
        first = config_module.ClassifierConfig(
            classifier_name='first',
            classes=['a'],
            beta=np.zeros((1024, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )
        second = config_module.ClassifierConfig(
            classifier_name='second',
            classes=['b'],
            beta=np.zeros((1024, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        configs = config_module.ClassifierConfigList(configs=[first, second])

        assert configs.embedding_dim == (1024,)

    def test_classifier_config_list_rejects_incompatible_embedding_dims(self):
        first = config_module.ClassifierConfig(
            classifier_name='first',
            classes=['a'],
            beta=np.zeros((1024, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )
        second = config_module.ClassifierConfig(
            classifier_name='second',
            classes=['b'],
            beta=np.zeros((1536, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        with pytest.raises(ValueError, match='same embedding_dim'):
            config_module.ClassifierConfigList(configs=[first, second])

    def test_classifier_config_embedding_model_name_perch_from_tfhub_v4(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a'],
            beta=np.zeros((10, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
            model_config={'model_config': {'tfhub_version': 4}},
        )

        assert config.embedding_model_name == 'perch_8'

    def test_classifier_config_embedding_model_name_perch_from_tfhub_v8(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a'],
            beta=np.zeros((10, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
            model_config={'model_config': {'tfhub_version': 8}},
        )

        assert config.embedding_model_name == 'perch_8'

    def test_classifier_config_embedding_model_name_perch_v2_from_shape(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a'],
            beta=np.zeros((1536, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
            model_config=None,
        )

        assert config.embedding_model_name == 'perch_v2'

    def test_classifier_config_embedding_model_name_birdnet_from_shape(self):
        config = config_module.ClassifierConfig(
            classifier_name='test_classifier',
            classes=['a'],
            beta=np.zeros((1024, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
            model_config=None,
        )

        assert config.embedding_model_name == 'birdnet_v2.4'

    def test_classifier_config_list_rejects_incompatible_embedding_model_names(self, monkeypatch):
        first = config_module.ClassifierConfig(
            classifier_name='first',
            classes=['a'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )
        second = config_module.ClassifierConfig(
            classifier_name='second',
            classes=['b'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        monkeypatch.setattr(
            config_module.ClassifierConfig,
            'embedding_model_name',
            property(lambda self: 'model_a' if self.classifier_name == 'first' else 'model_b'),
        )

        with pytest.raises(ValueError, match='same embedding_model_name or None'):
            config_module.ClassifierConfigList(configs=[first, second])

    def test_classifier_config_list_embedding_model_name_returns_unique_value(self, monkeypatch):
        first = config_module.ClassifierConfig(
            classifier_name='first',
            classes=['a'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )
        second = config_module.ClassifierConfig(
            classifier_name='second',
            classes=['b'],
            beta=np.zeros((1, 1), dtype=np.float32),
            beta_bias=np.zeros(1, dtype=np.float32),
        )

        monkeypatch.setattr(
            config_module.ClassifierConfig,
            'embedding_model_name',
            property(lambda self: 'shared_model'),
        )

        configs = config_module.ClassifierConfigList(configs=[first, second])

        assert configs.embedding_model_name == 'shared_model'
