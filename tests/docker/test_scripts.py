import subprocess
import json
from pathlib import Path
import pytest

from helpers import TestHelpers

import sys
scripts_module_path = Path(__file__).parent.parent.parent / 'scripts'
print(f"Adding scripts module path: {scripts_module_path}")
sys.path.insert(0, str(scripts_module_path))
import run_model_on_ecosounds

# The image name we'll use for testing
TEST_IMAGE_NAME = "crane-linear-model-runner:test"

#TODO: these should come from fixtures like the unit tests
TEST_ROOT = Path(__file__).parent.parent
INPUT_DIR = TEST_ROOT / "workspace" / "input"
OUTPUT_DIR = TEST_ROOT / "workspace" / "output"
CONFIG_DIR = TEST_ROOT / "workspace" / "config"


@pytest.mark.docker
def test_run_script(docker_image, clean_mounted_dirs):
    """
    Tests that the run_container.py script runs successfully
    and produces the expected output.
    """
    
    # Prepare input and config files
    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_parquet'), 
        dest_parent=INPUT_DIR,
        source_file='3757025.parquet',
        dest_file='test_input.parquet')

    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_config'), 
        dest_parent=CONFIG_DIR,
        source_file='config1.json',
        dest_file='test_config.json')

    # Run the script
    run_command = [
        "python", "scripts/run_container.py",
        "--input", str(INPUT_DIR / 'test_input.parquet'),
        "--output", str(OUTPUT_DIR / 'test_output.csv'),
        "--config", str(CONFIG_DIR / 'test_config.json')
    ]

    subprocess.run(run_command, check=True)

    # Check if the output file was created
    output_file = OUTPUT_DIR / 'test_output.csv'
    assert output_file.exists(), "Output file was not created!"
    assert output_file.stat().st_size > 0, "Output file is empty!"


@pytest.mark.ecosounds
def test_ecosounds_script_command(docker_image, clean_mounted_dirs):
    """
    Tests that the run_model_on_ecosounds.py script runs successfully
    and produces the expected output.
    
    Currently this will prompt for an API key and save it to an .env file
    Therefore this test should not be run automatically in CI.
    """

    dirs = clean_mounted_dirs
    
    # Prepare input and config files
    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_parquet'), 
        dest_parent=dirs['workspace_host'] / 'input',
        source_file='3757025.parquet',
        dest_file='ecosounds_input.parquet')

    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_config'), 
        dest_parent=dirs['workspace_host'] / 'config',
        source_file='config1.json',
        dest_file='ecosounds_config.json')
    
    
    params = {
        "filter": {"regions.id": {"eq": 116}},
        "recognizers": {
            "gbm_powerful_owl_03": str(dirs['workspace_host'] / 'config' / 'ecosounds_config.json')
        },
        "output": str(dirs['workspace_host'] / 'output')
        
    }
   
    params_file = dirs['workspace_host'] / 'ecosounds_script_params.json'
    with open(params_file, 'w') as f:
        json.dump(params, f)

    # Run the ecosounds script
    # This requires an actual connection to ecosounds with a valid API key
    run_command = [
        "python", "scripts/run_model_on_ecosounds.py",
        "--params", str(params_file),
        "--limit", "2"
    ]

    stdout, stderr = TestHelpers.sys_command(run_command)

    # Check if the output file was created
    output_file = dirs['workspace_host'] / 'output' / 'outputs/gbm_powerful_owl_03/gbm078_3761177/3761177.csv'
    assert output_file.exists(), "Ecosounds output file was not created!"
    assert output_file.stat().st_size > 0, "Ecosounds output file is empty!"


@pytest.mark.ecosounds
def test_ecosounds_script(docker_image, clean_mounted_dirs):
    """
    Tests that the run_model_on_ecosounds.py script runs successfully
    and produces the expected output.
    
    Currently this will prompt for an API key and save it to an .env file
    Therefore this test should not be run automatically in CI.
    """

    dirs = clean_mounted_dirs
    
    # Prepare input and config files
    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_parquet'), 
        dest_parent=dirs['workspace_host'] / 'input',
        source_file='3757025.parquet',
        dest_file='ecosounds_input.parquet')

    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_config'), 
        dest_parent=dirs['workspace_host'] / 'config',
        source_file='config1.json',
        dest_file='ecosounds_config.json')
    
    
    params = {
        "filter": {"regions.id": {"eq": 116}},
        "recognizers": {
            "gbm_powerful_owl_03": str(dirs['workspace_host'] / 'config' / 'ecosounds_config.json')
        },
        "output": str(dirs['workspace_host'] / 'output')
        
    }
   
    params_file = dirs['workspace_host'] / 'ecosounds_script_params.json'
    with open(params_file, 'w') as f:
        json.dump(params, f)


    run_model_on_ecosounds.main(params_file, limit=2)

    # Check if the output file was created
    output_file = dirs['workspace_host'] / 'output' / 'outputs/gbm_powerful_owl_03/gbm078_3761177/3761177.csv'
    assert output_file.exists(), "Ecosounds output file was not created!"
    assert output_file.stat().st_size > 0, "Ecosounds output file is empty!"


