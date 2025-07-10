#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration.
This file is automatically discovered by pytest.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json
import base64

from helpers import TestHelpers


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session") 
def test_data_dir(project_root):
    """Get the test data directory structure."""
    return {
        'files_parquet': project_root / 'files' / 'parquet',
        'files_config': project_root / 'files' / 'config',
        'mounted_input': project_root / 'mounted' / 'input',
        'mounted_config': project_root / 'mounted' / 'config',
        'mounted_output': project_root / 'mounted' / 'output'
    }


@pytest.fixture(scope="function")
def temp_workspace():
    """Create a temporary workspace for tests that don't use mounted dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        yield {
            'input': workspace / 'input',
            'config': workspace / 'config', 
            'output': workspace / 'output'
        }


@pytest.fixture(scope="function")
def sample_parquet_data():
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
def sample_config_data():
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


@pytest.fixture(scope="function", autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@pytest.fixture(scope="function")
def clean_mounted_dirs():
    """Fixture to clean mounted directories before each test."""
    dirs = TestHelpers.get_test_dirs()
    
    # Clean before test
    TestHelpers.clean_mounted_directory(dirs['mounted_input'])
    TestHelpers.clean_mounted_directory(dirs['mounted_config'])
    TestHelpers.clean_mounted_directory(dirs['mounted_output'])
    
    yield dirs
    
    # Clean after test (optional, but good practice)
    TestHelpers.clean_mounted_directory(dirs['mounted_input'])
    TestHelpers.clean_mounted_directory(dirs['mounted_config'])
    TestHelpers.clean_mounted_directory(dirs['mounted_output'])


@pytest.fixture(scope="function")
def sample_data(clean_mounted_dirs):
    """Fixture to create sample test data using PyArrow."""
    dirs = clean_mounted_dirs
    
    # Create sample parquet file
    parquet_path = dirs['mounted_input'] / 'test_data.parquet'
    sample_table = TestHelpers.create_sample_parquet(parquet_path, num_rows=50)
    
    # Create sample config
    config_path = dirs['mounted_config'] / 'test_config.json'
    config, beta, beta_bias = TestHelpers.create_sample_config(config_path)
    
    return {
        'parquet_path': str(parquet_path),
        'config_path': str(config_path),
        'output_dir': str(dirs['mounted_output']),
        'sample_table': sample_table,
        'config': config,
        'beta': beta,
        'beta_bias': beta_bias
    }


@pytest.fixture
def real_data_file(request, clean_mounted_dirs):
    """
    A parametrized fixture that sets up real files based on parameters
    passed from the test function.
    """
   
    input_filename = request.param[0]
    config_filename = request.param[1]
    output_filename = request.param[2] if len(request.param) > 2 else None

    if output_filename is None:
        input_filename = Path(input_filename)
        output_filename = input_filename.with_name(f"{input_filename.stem}.results.csv")

    TestHelpers.copy_input(input_filename, input_filename)
    TestHelpers.copy_config(config_filename, config_filename)

    dirs = TestHelpers.get_test_dirs()

    return {
        'input_path': dirs['mounted_input'] / input_filename,
        'config_path': dirs['mounted_config'] / config_filename,
        'output_path': dirs['mounted_output'] / output_filename
    }


@pytest.fixture
def real_data_folder(request, clean_mounted_dirs):
    """
    A parametrized fixture that sets up real files based on parameters
    passed from the test function.
    """
   
    input_filenames = request.param[0]
    config_filename = request.param[1]

    if isinstance(input_filenames, str):
        input_filenames = [input_filenames]

    for input_filename in input_filenames:
        TestHelpers.copy_input(input_filename, input_filename)
    TestHelpers.copy_config(config_filename, config_filename)

    dirs = TestHelpers.get_test_dirs()

    return {
        'input_path': dirs['mounted_input'],
        'config_path': dirs['mounted_config'],
        'output_path': dirs['mounted_output']
    }



# Custom markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test" 
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(mark.name in ['integration', 'slow'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
            
        # Add slow marker to tests that take longer
        if 'directory_processing' in item.name or 'large' in item.name:
            item.add_marker(pytest.mark.slow)