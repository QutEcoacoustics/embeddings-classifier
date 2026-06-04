
import json
import copy
from pathlib import Path
import pytest

import numpy as np


import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import TestHelpers
from unit_helpers import UnitTestHelpers
import embeddings_classifier.app as app

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
        beta, beta_bias = app.deserialize_classifier_params(classifier)
        
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
            app.deserialize_classifier_params(classifier)
    

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
        config_list = app.ClassifierConfig.from_any(sample_data['config_path']).as_list()
        second = dict(config_list[0])
        second['classifier_name'] = 'A_B'
        config_list.append(second)
        config_list[0]['classifier_name'] = 'A B'
        configs = app.ClassifierConfig(configs=config_list)

        with pytest.raises(ValueError, match='resolve to the same output path'):
            app.init_items(configs, Path(sample_data['output_dir']) / '<classifier_name>' / 'result.csv')

    def test_resolve_classifier_name_uses_name_when_classifier_name_missing(self):
        """The generic 'name' key should be accepted as classifier name fallback."""
        config = {
            'name': 'Powerful Owl',
            'classifier': {'classes': ['owl']},
        }
        assert app.resolve_classifier_name(config, 0) == 'Powerful Owl'

    def test_resolve_classifier_name_single_class_fallback(self):
        """Single-class classifiers should derive name from class label."""
        config = {
            'classifier': {'classes': ['Yellow bellied glider']},
        }
        assert app.resolve_classifier_name(config, 2) == 'yellow_bellied_glider'

    def test_resolve_classifier_name_multi_class_fallback(self):
        """Multi-class classifiers should use informative short fallback names."""
        config = {
            'classifier': {'classes': ['a', 'b', 'c']},
        }
        assert app.resolve_classifier_name(config, 1) == 'classifier_1_3class'

    def test_init_items_raises_on_derived_name_collision(self, sample_data):
        """Collisions from class-derived fallback names should be detected."""
        config_list = app.ClassifierConfig.from_any(sample_data['config_path']).as_list()
        base = dict(config_list[0])
        base.pop('classifier_name', None)
        base.pop('name', None)
        base['classifier'] = dict(base['classifier'])
        base['classifier']['classes'] = ['A B']

        second = dict(base)
        second['classifier'] = dict(second['classifier'])
        second['classifier']['classes'] = ['a_b']

        base['classifier_name'] = app.resolve_classifier_name(base, 0)
        second['classifier_name'] = app.resolve_classifier_name(second, 1)

        configs = app.ClassifierConfig(configs=[base, second])

        with pytest.raises(ValueError, match='resolve to the same output path'):
            app.init_items(configs, Path(sample_data['output_dir']) / '<classifier_name>' / 'result.csv')

    def test_build_threshold_array_scalar(self):
        classes = ['a', 'b']
        result = app.build_threshold_array(classes, 0.25)
        np.testing.assert_allclose(result, np.array([0.25, 0.25], dtype=np.float32))

    def test_build_threshold_array_none(self):
        classes = ['a', 'b']
        result = app.build_threshold_array(classes, None)
        expected = np.array([np.finfo(np.float32).min, np.finfo(np.float32).min], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_build_threshold_array_list_and_length_validation(self):
        classes = ['a', 'b']
        result = app.build_threshold_array(classes, [0.1, None])
        expected = np.array([0.1, np.finfo(np.float32).min], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

        with pytest.raises(ValueError, match='Threshold list length'):
            app.build_threshold_array(classes, [0.1])

    def test_build_threshold_array_dict_with_missing_and_none(self):
        classes = ['a', 'b', 'c']
        result = app.build_threshold_array(classes, {'a': 0.2, 'b': None})
        expected = np.array([0.2, np.finfo(np.float32).min, 0.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_build_threshold_array_invalid_types(self):
        classes = ['a']

        with pytest.raises(TypeError, match='Threshold must be one of'):
            app.build_threshold_array(classes, 'bad')

        with pytest.raises(TypeError, match="Threshold for class 'a' must be a number or None"):
            app.build_threshold_array(classes, {'a': 'bad'})

    def test_from_any_does_not_mutate_input_with_threshold_array(self, sample_data):
        raw = copy.deepcopy(sample_data['config'])

        assert 'threshold_array' not in raw

        normalized = app.ClassifierConfig.from_any(raw).as_list()[0]

        assert 'threshold_array' in normalized
        assert isinstance(normalized['threshold_array'], np.ndarray)
        assert 'threshold_array' not in raw
