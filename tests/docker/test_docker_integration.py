import subprocess
import os
import shutil
from pathlib import Path
import pytest

from helpers import TestHelpers

# The image name we'll use for testing
TEST_IMAGE_NAME = "crane-linear-model-runner:test"

#TODO: these should come from fixtures like the unit tests
TEST_ROOT = Path(__file__).parent.parent
INPUT_DIR = TEST_ROOT / "workspace" / "input"
OUTPUT_DIR = TEST_ROOT / "workspace" / "output"
CONFIG_DIR = TEST_ROOT / "workspace" / "config"

@pytest.mark.docker
def test_docker_build_creates_image():
    """
    Tests that the Docker image builds successfully and exists locally.
    """

    TestHelpers.sys_command(["docker", "build", "-t", TEST_IMAGE_NAME, "."])

    stdout, stderr = TestHelpers.sys_command(["docker", "image", "inspect", TEST_IMAGE_NAME])

    # If the code reaches this point, the image was successfully built and found.
    # The assertion below is a final sanity check.
    assert stdout, "docker image inspect should produce output for an existing image."
    

@pytest.mark.docker
def test_docker_run_produces_output_with_params(docker_image, clean_mounted_dirs):
    """
    Tests that running the container produces the expected output file.
    This test depends on the image already being built.
    """

    # copies test_data into the workspace that gets mounted into the container
    # copy_test_file(source_parent, dest_parent, source_file, dest_file = None):

    # we mount just 1 directory to a custom location in the container, then pass as parameters
    host_folder = TestHelpers.get_test_dirs('workspace_input')

    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_config'), 
        dest_parent=host_folder,
        source_file='config1.json',
        dest_file='my_config.json')
    
    TestHelpers.copy_test_file(
        source_parent=TestHelpers.get_test_dirs('data_parquet'), 
        dest_parent=host_folder,
        source_file='3757025.parquet',
        dest_file='my_embedding.parquet')

    custom_mount = "/mnt/somewhere"
    
    run_command = [
        "docker", "run", "--rm",
        "-v", f"{host_folder.resolve()}:{custom_mount}",
        docker_image, 
        "classify", 
        "--input", f"{custom_mount}/my_embedding.parquet",
        "--output", f"{custom_mount}/hi_mum.csv",
        "--config", f"{custom_mount}/my_config.json"
    ]

    subprocess.run(run_command, check=True)
    
    expected_output_file = host_folder / 'hi_mum.csv' 
    assert expected_output_file.exists(), "Output file was not created!"
    assert expected_output_file.stat().st_size > 0, "Output file is empty!"


@pytest.mark.docker
def test_docker_run_uses_default_paths(docker_image, clean_mounted_dirs):
    """
    Tests that `docker run` with no arguments correctly uses the
    default /app/input, /app/output, and /app/config paths.
    """
    TestHelpers.copy_config('config1.json', 'config.json')
    TestHelpers.copy_input('3757025.parquet')

    run_command = [
        "docker", "run", "--rm",
        "-v", f"{CONFIG_DIR.resolve()}:/mnt/config",
        "-v", f"{INPUT_DIR.resolve()}:/mnt/input",
        "-v", f"{OUTPUT_DIR.resolve()}:/mnt/output",
        docker_image
    ]
    
    stdout, stderr = TestHelpers.sys_command(run_command)

    expected_output_file = OUTPUT_DIR / "3757025.csv" 
    assert expected_output_file.exists(), "Output file was not created in the default location!"

