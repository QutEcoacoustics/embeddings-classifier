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

import sys
sys.path.insert(0, str(Path(__file__).parent))

from helpers import TestHelpers

# @pytest.fixture(scope="session")
# def project_root():
#     """Get the project root directory."""
#     return Path(__file__).parent.parent


# @pytest.fixture(scope="session")
# def pa():
#     """A fixture that lazily imports and returns the pyarrow module."""
#     import pyarrow
#     return pyarrow


# @pytest.fixture(scope="session")
# def pq(pa):
#     """A fixture that lazily imports and returns pyarrow.parquet."""
#     import pyarrow.parquet
#     return pyarrow.parquet


# @pytest.fixture(scope="session")
# def np():
#     """A fixture that lazily imports and returns numpy."""
#     import numpy
#     return numpy


@pytest.fixture(scope="session") 
def test_data_dir(project_root):
    """Get the test data directory structure."""

    return TestHelpers.get_test_dirs()



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
    TestHelpers.clean_mounted_directory(dirs['workspace_input'])
    TestHelpers.clean_mounted_directory(dirs['workspace_config'])
    TestHelpers.clean_mounted_directory(dirs['workspace_output'])
    TestHelpers.clean_mounted_directory(dirs['workspace_host'])
    
    yield dirs
    
    # clean after test
    TestHelpers.clean_mounted_directory(dirs['workspace_input'])
    TestHelpers.clean_mounted_directory(dirs['workspace_config'])
    TestHelpers.clean_mounted_directory(dirs['workspace_output'])
    TestHelpers.clean_mounted_directory(dirs['workspace_host'])



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
        'input_path': dirs['workspace_input'] / input_filename,
        'config_path': dirs['workspace_config'] / config_filename,
        'output_path': dirs['workspace_output'] / output_filename
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
        'input_path': dirs['workspace_input'],
        'config_path': dirs['workspace_config'],
        'output_path': dirs['workspace_output']
    }



def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(mark.name in ['integration', 'slow'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
            
        # Add slow marker to tests that take longer
        if 'directory_processing' in item.name or 'large' in item.name:
            item.add_marker(pytest.mark.slow)