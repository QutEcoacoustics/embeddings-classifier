#!/usr/bin/env python3
"""
Test suite for the parquet processor main function using pytest.
"""

import json
import os
from pathlib import Path
import re
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as pv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import TestHelpers
from unit_helpers import UnitTestHelpers
import embeddings_classifier.app as app
import embeddings_classifier.config as config_module


def _first_classifier_folder(config_path):
    configs = list(config_module.ClassifierConfigList.from_any(config_path))
    classifier_name = configs[0].classifier_name
    classifier_name = re.sub(r'\s+', '_', classifier_name)
    classifier_name = re.sub(r'[^A-Za-z0-9_-]', '', classifier_name)
    return classifier_name


class TestClassifyFunction:
    """Test cases for the main function."""

    # grtbgw_inputs = [('20230324T090000+1100_Kl9_3372829.wav.parquet', 'grtbgw.json')]

    # @pytest.mark.parametrize('real_data_file', grtbgw_inputs, indirect=True)
    # def test_real_file_processing(self, real_data_file):
    #     """
    #     Tests single file processing using a parametrized fixture to run
    #     multiple file combinations through the same test logic.
    #     """
                
    #     app.classify(real_data_file['input_path'], 
    #                  real_data_file['output_path'], 
    #                  real_data_file['config_path'])
        
    #     assert Path(real_data_file['output_path']).exists()

    #     UnitTestHelpers.compare_predictions(
    #         real_data_file['output_path'],
    #         './tests/test_data/predictions/grtbgw.csv')
        


    def test_real_file_processing(self, clean_mounted_dirs):
        """
        Tests single file processing using a parametrized fixture to run
        multiple file combinations through the same test logic.
        """

        dirs = clean_mounted_dirs

        input_file = TestHelpers.copy_input('20230324T090000+1100_Kl9_3372829.wav.parquet')
        config_file = TestHelpers.copy_config('grtbgw.json')


        app.classify(input_file, 
                     dirs['workspace_output'], 
                     config_file)

        classifier_folder = _first_classifier_folder(config_file)
        
        expected_output_file = dirs['workspace_output'] / classifier_folder / '20230324T090000+1100_Kl9_3372829.wav.csv'
        
        assert Path(expected_output_file).exists()

        UnitTestHelpers.compare_predictions(
            expected_output_file,
            './tests/test_data/predictions/grtbgw.csv')

    
    def test_single_file_processing(self, sample_data):
        """Test processing a single parquet file."""
        input_path = Path(sample_data['parquet_path'])
        output_path = Path(sample_data['output_dir']) / 'output.parquet'
        config_path = Path(sample_data['config_path'])
        classifier_folder = _first_classifier_folder(config_path)
        expected_output_path = Path(sample_data['output_dir']) / classifier_folder / 'output.parquet'

        app.classify(input_path, output_path, config_path)
        
        # Verify output file exists
        assert os.path.exists(expected_output_path)
        
        # Load and verify output using PyArrow
        result_table = pq.read_table(expected_output_path)
        
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

        classifier_folder = _first_classifier_folder(config_path)

        expected_output_files = [
            output_dir / classifier_folder / 'file1.csv',
            output_dir / classifier_folder / 'subdir1' / 'file2.csv',
            output_dir / classifier_folder / 'subdir2' / 'file3.csv'
        ]

        for expected_output_file in expected_output_files:
            assert expected_output_file.exists()
            result_table = pv.read_csv(expected_output_file)
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
        
        with pytest.raises(FileNotFoundError):
            app.classify(Path('/nonexistent/path'), output_path, config_path)
    

    def test_invalid_config_path(self, sample_data):
        """Test handling of invalid config path."""
        input_path = sample_data['parquet_path']
        output_path = Path(sample_data['output_dir']) / 'output.parquet'
        
        with pytest.raises(FileNotFoundError):
            app.classify(input_path, output_path, '/nonexistent/config.json')
    

    def test_empty_directory(self, clean_mounted_dirs):
        """Test handling of empty input directory."""
        dirs = clean_mounted_dirs
        
        input_dir = dirs['workspace_input']
        config_path = dirs['workspace_config'] / 'test_config.json'
        output_dir = dirs['workspace_output']
        
        # Create config but no parquet files
        UnitTestHelpers.create_sample_config(config_path)
        
        with pytest.raises(RuntimeError):
            app.classify(input_dir, output_dir, config_path)
    

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
        
        with pytest.raises(RuntimeError):
            app.classify(parquet_path, output_path, config_path)
        
        assert not os.path.exists(output_path)
    



    # test_cases = [
    #     ('3757025.parquet', 'config1.json'),
    #     ('3757025.parquet', 'config_classifier_only.json')
    # ]

    # @pytest.mark.parametrize('real_data_file', test_cases, indirect=True)
    def test_single_file_processing_real_fixture(self, clean_mounted_dirs):

        #TODO: we might already have an equivalent test in test_real_file_processing

        dirs = clean_mounted_dirs

        input_file = TestHelpers.copy_input('3757025.parquet')
        config_file = TestHelpers.copy_config('config1.json')

 
        app.classify(input_file,
                dirs['workspace_output'],
                config_file)

        classifier_folder = _first_classifier_folder(config_file)
        
        expected_output_path = dirs['workspace_output'] / classifier_folder / '3757025.csv'
        
        assert Path(expected_output_path).exists()
        result_table = pv.read_csv(expected_output_path)
        assert 'score' in result_table.column_names


    def test_single_real_file_bare_classifier(self, clean_mounted_dirs):
        """
        Tests that config with only the classifier is processed correctly.
        Config should be normalized to standard format with defaults applied.
        """

        dirs = clean_mounted_dirs

        input_file = TestHelpers.copy_input('3757025.parquet')
        config_file = TestHelpers.copy_config('config_classifier_only.json')

 
        app.classify(input_file,
                dirs['workspace_output'],
                config_file)

        classifier_folder = _first_classifier_folder(config_file)
        
        expected_output_path = dirs['workspace_output'] / classifier_folder / '3757025.csv'
        
        assert Path(expected_output_path).exists()
        result_table = pv.read_csv(expected_output_path)
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

        classifier_folder = _first_classifier_folder(config_path)
        
        assert (output_dir / classifier_folder / '3757025.csv').exists()
        assert (output_dir / classifier_folder / 'group_a' / 'real_file_2.csv').exists()


    @pytest.mark.auth_integration
    def test_url_processing(self, clean_mounted_dirs, monkeypatch, validated_baw_auth_token):
        """Test processing a directory containing copies of the real parquet file."""

        monkeypatch.setenv('QSP', f'user_token={validated_baw_auth_token}')

        TestHelpers.copy_config('config1.json')
        TestHelpers.copy_input('url_list_1.json')
        
        dirs = clean_mounted_dirs
        input_json_file = dirs['workspace_input'] / 'url_list_1.json'
        output_dir = dirs['workspace_output']
        config_path = dirs['workspace_config'] / 'config1.json'
  
        app.classify(input_json_file, output_dir, config_path)

        classifier_folder = _first_classifier_folder(config_path)
        
        assert (output_dir / classifier_folder / 'file_1' / '3809284_detections.csv').exists()
        assert (output_dir / classifier_folder / 'file_2' / '3809700_detections.csv').exists()


    @pytest.mark.skip(reason="TODO: add a redacted/synthetic large URL-manifest integration test")
    def test_url_processing_large_manifest_todo(self):
        """
        Placeholder for validating large JSON URL input manifests.

        The active test_url_processing already covers URL JSON parsing and auth.
        This TODO should focus on scale/shape with safe non-sensitive fixture data.
        """
        pass

    def test_classify_table_in_memory_only(self, sample_data):
        """Classify a preloaded Arrow table without writing output files."""
        table = sample_data['sample_table']
        config_path = sample_data['config_path']
        output_dir = Path(sample_data['output_dir'])

        results = app.classify_table(table, config_path)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].result_table is not None
        assert results[0].output_path is None
        assert list(output_dir.rglob('*')) == []

    def test_classify_table_with_output_path(self, sample_data):
        """Classify a preloaded Arrow table and write outputs using classifier path templating."""
        table = sample_data['sample_table']
        config_path = sample_data['config_path']
        output_base = Path(sample_data['output_dir']) / 'result.parquet'
        classifier_folder = _first_classifier_folder(config_path)
        expected_output = Path(sample_data['output_dir']) / classifier_folder / 'result.parquet'

        results = app.classify_table(table, config_path, output_path=output_base)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].result_table is not None
        assert results[0].output_path == expected_output
        assert expected_output.exists()

    def test_full_output_path_templates_absolute_path_ignores_parent_token(self):
        """Absolute output paths should inject token unless the absolute path already has one."""
        absolute_output = Path('/tmp/out/result.csv')
        output_parent = Path('/tmp/<classifier_name>/base')

        templates = app.get_full_output_path_templates(
            [absolute_output],
            output_parent,
            [Path('/tmp/input.parquet')],
        )

        assert templates == [Path('/tmp/out/<classifier_name>/result.csv')]

    def test_process_single_input_in_memory_only(self, sample_data):
        """Shared single-input path should support no-write mode."""
        table = sample_data['sample_table']
        configs = config_module.ClassifierConfigList.from_any(sample_data['config_path'])

        results = app._process_single_input(table, None, configs)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output_path is None
        assert results[0].result_table is not None

    def test_classify_continues_after_unexpected_file_error_single_worker(self, clean_mounted_dirs, monkeypatch):
        """An unexpected exception for one file should not stop remaining files in serial mode."""
        dirs = clean_mounted_dirs

        input_dir = dirs['workspace_input']
        output_dir = dirs['workspace_output']
        config_path = dirs['workspace_config'] / 'test_config.json'

        UnitTestHelpers.create_sample_parquet(input_dir / 'crash.parquet', num_rows=5)
        UnitTestHelpers.create_sample_parquet(input_dir / 'ok.parquet', num_rows=5)
        UnitTestHelpers.create_sample_config(config_path)

        seen_inputs = []

        def fake_process_single_input(input_path, output_path_template, configs):
            seen_inputs.append(Path(input_path).name)
            if Path(input_path).name == 'crash.parquet':
                raise AssertionError('unexpected crash')
            return [SimpleNamespace(success=True)]

        monkeypatch.setattr(app, '_process_single_input', fake_process_single_input)

        with pytest.raises(RuntimeError, match='Encountered 1 error\(s\) during processing'):
            app.classify(input_dir, output_dir, config_path, workers=1)

        assert sorted(seen_inputs) == ['crash.parquet', 'ok.parquet']

    def test_classify_continues_after_unexpected_file_error_multi_worker(self, clean_mounted_dirs, monkeypatch):
        """An unexpected exception for one file should not stop remaining files in parallel mode."""
        dirs = clean_mounted_dirs

        input_dir = dirs['workspace_input']
        output_dir = dirs['workspace_output']
        config_path = dirs['workspace_config'] / 'test_config.json'

        UnitTestHelpers.create_sample_parquet(input_dir / 'crash.parquet', num_rows=5)
        UnitTestHelpers.create_sample_parquet(input_dir / 'ok.parquet', num_rows=5)
        UnitTestHelpers.create_sample_config(config_path)

        seen_inputs = []

        def fake_process_single_input(input_path, output_path_template, configs):
            seen_inputs.append(Path(input_path).name)
            if Path(input_path).name == 'crash.parquet':
                raise AssertionError('unexpected crash')
            return [SimpleNamespace(success=True)]

        monkeypatch.setattr(app, '_process_single_input', fake_process_single_input)

        with pytest.raises(RuntimeError, match='Encountered 1 error\(s\) during processing'):
            app.classify(input_dir, output_dir, config_path, workers=2)

        assert sorted(seen_inputs) == ['crash.parquet', 'ok.parquet']

    def test_classify_dataframe_matches_classify_table(self, sample_data):
        """DataFrame entrypoint should match classify_table output shape and rows."""
        pd = pytest.importorskip("pandas")

        table = sample_data['sample_table']
        df = table.to_pandas()
        config_path = sample_data['config_path']

        table_results = app.classify_table(table, config_path)
        dataframe_results = app.classify_dataframe(df, config_path)

        assert len(table_results) == len(dataframe_results) == 1
        assert dataframe_results[0].success is True
        assert dataframe_results[0].result_table is not None
        assert table_results[0].result_table is not None
        assert dataframe_results[0].result_table.num_rows == table_results[0].result_table.num_rows
        assert dataframe_results[0].result_table.column_names == table_results[0].result_table.column_names


if __name__ == '__main__':
    pytest.main([__file__])