#!/usr/bin/env python3
"""
Test suite for the parquet processor main function using pytest.
"""

import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as pv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import TestHelpers
from unit_helpers import UnitTestHelpers
import app 


class TestClassifyFunction:
    """Test cases for the main function."""

    grtbgw_inputs = [('20230324T090000+1100_Kl9_3372829.wav.parquet', 'grtbgw.json')]

    @pytest.mark.parametrize('real_data_file', grtbgw_inputs, indirect=True)
    def test_real_file_processing(self, real_data_file):
        """
        Tests single file processing using a parametrized fixture to run
        multiple file combinations through the same test logic.
        """
                
        app.classify(real_data_file['input_path'], 
                     real_data_file['output_path'], 
                     real_data_file['config_path'])
        
        assert Path(real_data_file['output_path']).exists()

        UnitTestHelpers.compare_predictions(
            real_data_file['output_path'],
            './tests/test_data/predictions/grtbgw.csv')

    
    def test_single_file_processing(self, sample_data):
        """Test processing a single parquet file."""
        input_path = Path(sample_data['parquet_path'])
        output_path = Path(sample_data['output_dir']) / 'output.parquet'
        config_path = Path(sample_data['config_path'])
        
        app.classify(input_path, output_path, config_path)
        
        # Verify output file exists
        assert os.path.exists(output_path)
        
        # Load and verify output using PyArrow
        result_table = pq.read_table(output_path)
        
        # Check structure
        assert 'score' in result_table.column_names
        #assert result_table.num_rows == sample_data['sample_table'].num_rows

        # metadata columns preserved
        assert set(result_table.column_names[0:3]) == set(sample_data['sample_table'].column_names[0:3])  
        
        # Verify scores are reasonable (not NaN, finite)
        scores = result_table.column('score').to_numpy()
        assert not np.isnan(scores).any()
        assert np.isfinite(scores).all()
    

    def test_directory_processing(self, clean_mounted_dirs):
        """Test processing a directory of parquet files."""
        dirs = clean_mounted_dirs
        

        input_dir = dirs['workspace_input']
        config_path = dirs['workspace_config'] / 'test_config.json'
        output_dir = dirs['workspace_output']
        
        (input_dir / 'subdir1').mkdir(parents=True, exist_ok=True)
        (input_dir / 'subdir2').mkdir(parents=True, exist_ok=True)
        
        UnitTestHelpers.create_sample_parquet(input_dir / 'file1.parquet', num_rows=30)
        UnitTestHelpers.create_sample_parquet(input_dir / 'subdir1' / 'file2.parquet', num_rows=40)
        UnitTestHelpers.create_sample_parquet(input_dir / 'subdir2' / 'file3.parquet', num_rows=25)
        UnitTestHelpers.create_sample_config(config_path)
    
        app.classify(input_dir, output_dir, config_path)
        
        assert (output_dir / 'file1.csv').exists()
        assert (output_dir / 'subdir1' / 'file2.csv').exists()
        assert (output_dir / 'subdir2' / 'file3.csv').exists()
        
        for output_file in [
            output_dir / 'file1.csv',
            output_dir / 'subdir1' / 'file2.csv',
            output_dir / 'subdir2' / 'file3.csv'
        ]:
            result_table = pv.read_csv(output_file)
            assert 'score' in result_table.column_names
            scores = result_table.column('score').to_numpy()
            assert not np.isnan(scores).any()

    def test_threshold(self, clean_mounted_dirs):
        #TODO: Implement test for threshold functionality
        pass

        # TestHelpers.copy_config('test_config2.json')

        

    

    def test_invalid_input_path(self, sample_data):
        """Test handling of invalid input path."""
        config_path = sample_data['config_path']
        output_path = Path(sample_data['output_dir']) / 'output.parquet'
        
        with pytest.raises(SystemExit) as exc_info:
            app.classify(Path('/nonexistent/path'), output_path, config_path)
        assert exc_info.value.code == 1
    

    def test_invalid_config_path(self, sample_data):
        """Test handling of invalid config path."""
        input_path = sample_data['parquet_path']
        output_path = Path(sample_data['output_dir']) / 'output.parquet'
        
        with pytest.raises(SystemExit) as exc_info:
            app.classify(input_path, output_path, '/nonexistent/config.json')
        assert exc_info.value.code == 1
    

    def test_empty_directory(self, clean_mounted_dirs):
        """Test handling of empty input directory."""
        dirs = clean_mounted_dirs
        
        input_dir = dirs['workspace_input']
        config_path = dirs['workspace_config'] / 'test_config.json'
        output_dir = dirs['workspace_output']
        
        # Create config but no parquet files
        UnitTestHelpers.create_sample_config(config_path)
        
        with pytest.raises(SystemExit) as exc_info:
            app.classify(input_dir, output_dir, config_path)
        assert exc_info.value.code == 1
    

    def test_mismatched_feature_dimensions(self, clean_mounted_dirs):
        """Test handling of mismatched feature dimensions."""
        dirs = clean_mounted_dirs
        
        # Create parquet with different number of features
        parquet_path = dirs['workspace_input'] / 'test_data.parquet'
        UnitTestHelpers.create_sample_parquet(parquet_path, num_rows=30, num_features=500)  
        
        # Create config with 1280 features
        config_path = dirs['workspace_config'] / 'test_config.json'
        UnitTestHelpers.create_sample_config(config_path, num_features=1280)
        
        output_path = dirs['workspace_output'] / 'output.parquet'
        
        with pytest.raises(SystemExit) as exc_info:
            app.classify(parquet_path, output_path, config_path)
        assert exc_info.value.code == 1
        
        assert not os.path.exists(output_path)
    



    test_cases = [
        ('3757025.parquet', 'config1.json'),
        ('3757025.parquet', 'config_classifier_only.json')
    ]

    @pytest.mark.parametrize('real_data_file', test_cases, indirect=True)
    def test_single_file_processing_parametrized(self, real_data_file):
        """
        Tests single file processing using a parametrized fixture to run
        multiple file combinations through the same test logic.
        """
 
        app.classify(real_data_file['input_path'],
                real_data_file['output_path'],
                real_data_file['config_path'])
        
        assert Path(real_data_file['output_path']).exists()
        result_table = pv.read_csv(real_data_file['output_path'])
        assert 'score' in result_table.column_names
            

    def test_directory_processing_with_real_files(self, clean_mounted_dirs):
        """Test processing a directory containing copies of the real parquet file."""

        TestHelpers.copy_config('config1.json')
        TestHelpers.copy_input('3757025.parquet')
        TestHelpers.copy_input('3757025.parquet', Path('group_a') / 'real_file_2.parquet')
        
        dirs = TestHelpers.get_test_dirs()
        input_dir = dirs['workspace_input']
        output_dir = dirs['workspace_output']
        config_path = dirs['workspace_config'] / 'config1.json'
  
        app.classify(input_dir, output_dir, config_path)
        
        assert (output_dir / '3757025.csv').exists()
        assert (output_dir / 'group_a' / 'real_file_2.csv').exists()



if __name__ == '__main__':
    pytest.main([__file__])