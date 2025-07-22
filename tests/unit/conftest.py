#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration.
This file is automatically discovered by pytest.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import base64

from unit_helpers import UnitTestHelpers

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# @pytest.fixture(scope="session")
# def project_root():
#     """Get the project root directory."""
#     return Path(__file__).parent.parent


# @pytest.fixture(scope="session")
# def pa():
#     """A fixture that lazily imports and returns the pyarrow module."""

#     return pyarrow


# @pytest.fixture(scope="session")
# def pq(pa):
#     """A fixture that lazily imports and returns pyarrow.parquet."""

#     return pyarrow.parquet


# @pytest.fixture(scope="session")
# def np():
#     """A fixture that lazily imports and returns numpy."""

#     return numpy



@pytest.fixture(scope="function")
def sample_parquet_data(pa, np):
    """Generate sample parquet data for testing using PyArrow."""


    num_rows = 100
    num_features = 1280
    
    # Set seed for reproducible tests
    np.random.seed(42)
    
    # Create arrays
    id_array = pa.array(range(num_rows))
    label_array = pa.array([f'item_{i}' for i in range(num_rows)])
    
    # Add feature columns
    feature_arrays = []
    feature_names = []
    for i in range(num_features):
        feature_data = np.random.randn(num_rows).astype(np.float32)
        feature_arrays.append(pa.array(feature_data))
        feature_names.append(f'feature_{i}')
    
    # Combine all arrays and names
    all_arrays = [id_array, label_array] + feature_arrays
    all_names = ['id', 'label'] + feature_names
    
    return pa.table(all_arrays, names=all_names)


@pytest.fixture(scope="function")
def sample_config_data(np):
    """Generate sample configuration data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    num_features = 1280
    beta = np.random.randn(num_features).astype(np.float32)
    beta_bias = np.random.randn(1).astype(np.float32)
    
    # Encode to base64
    beta_b64 = base64.b64encode(beta.tobytes()).decode('utf-8')
    beta_bias_b64 = base64.b64encode(beta_bias.tobytes()).decode('utf-8')
    
    config = {
        "classifier": {
            "classes": ["Yellow-bellied glider"],
            "beta": beta_b64,
            "beta_bias": beta_bias_b64
        },
        "input_type": "test"
    }
    
    return config, beta, beta_bias






@pytest.fixture(scope="function")
def sample_data(clean_mounted_dirs):
    """Fixture to create sample test data using PyArrow."""
    dirs = clean_mounted_dirs
    
    # Create sample parquet file
    parquet_path = dirs['workspace_input'] / 'test_data.parquet'
    sample_table = UnitTestHelpers.create_sample_parquet(parquet_path, num_rows=50)
    
    # Create sample config
    config_path = dirs['workspace_config'] / 'test_config.json'
    config, beta, beta_bias = UnitTestHelpers.create_sample_config(config_path)
    
    return {
        'parquet_path': str(parquet_path),
        'config_path': str(config_path),
        'output_dir': str(dirs['workspace_output']),
        'sample_table': sample_table,
        'config': config,
        'beta': beta,
        'beta_bias': beta_bias
    }



def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests that are not marked as docker
        if not any(mark.name in ['docker'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
            
