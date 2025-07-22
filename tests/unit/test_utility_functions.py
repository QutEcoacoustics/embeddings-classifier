
import json
from pathlib import Path
import pytest

import numpy as np


import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import TestHelpers
from unit_helpers import UnitTestHelpers
import app 

class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_load_config_valid(self, sample_data):
        """Test loading valid configuration."""
        config = app.load_config(sample_data['config_path'])
        
        assert 'classifier' in config
        assert 'classes' in config['classifier']
        assert 'beta' in config['classifier']
        assert 'beta_bias' in config['classifier']
    
    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON configuration."""
        config_path = tmp_path / "invalid_config.json"
        config_path.write_text("{ invalid json }")
        
        with pytest.raises(SystemExit) as exc_info:
            app.load_config(str(config_path))
        assert exc_info.value.code == 1
    

    def test_load_config_missing_classifier(self, tmp_path):
        """Test loading config missing classifier section."""
        config_path = tmp_path / "no_classifier.json"
        config_path.write_text('{"other": "data"}')
        
        with pytest.raises(SystemExit) as exc_info:
            app.load_config(str(config_path))
        assert exc_info.value.code == 1
    

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
        
        with pytest.raises(SystemExit) as exc_info:
            app.load_config(str(config_path))
        assert exc_info.value.code == 1
    

    def test_deserialize_classifier_params(self, sample_data):
        """Test deserializing classifier parameters."""
        config = app.load_config(sample_data['config_path'])
        beta, beta_bias = app.deserialize_classifier_params(config['classifier'])
        
        assert isinstance(beta, np.ndarray)
        assert isinstance(beta_bias, np.ndarray)
        assert beta.dtype == np.float32
        assert beta_bias.dtype == np.float32
    

    def test_deserialize_classifier_params_invalid_base64(self, tmp_path):
        """Test deserializing invalid base64 data."""
        config_path = tmp_path / "bad_base64_config.json"
        config_data = {
            "classifier": {
                "classes": ["test"],
                "beta": "invalid_base64!@#",
                "beta_bias": "also_invalid!@#"
            }
        }
        config_path.write_text(json.dumps(config_data))
        
        config = app.load_config(str(config_path))
        
        with pytest.raises(SystemExit) as exc_info:
            app.deserialize_classifier_params(config['classifier'])
        assert exc_info.value.code == 1
    

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
