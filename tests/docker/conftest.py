#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration.
This file is automatically discovered by pytest.
"""

import pytest
from pathlib import Path

from helpers import TestHelpers 

TEST_IMAGE_NAME = "crane-linear-model-runner:test"

@pytest.fixture(scope="session")
def docker_image():
    """
    Builds the Docker image once for the test session and yields its name.
    """
    print(f"\nBuilding Docker image: {TEST_IMAGE_NAME}")
    
    # The helper's `check=True` will raise an exception if the build fails.
    TestHelpers.sys_command(["docker", "build", "-t", TEST_IMAGE_NAME, "."])
    
    # 'yield' passes the image name to the tests. The code below this
    # line will run after all tests in the session are complete.
    yield TEST_IMAGE_NAME
    
    # Optional: Cleanup by removing the test image after the session
    print(f"\nCleaning up Docker image: {TEST_IMAGE_NAME}")
    TestHelpers.sys_command(["docker", "image", "rm", "-f", TEST_IMAGE_NAME])
