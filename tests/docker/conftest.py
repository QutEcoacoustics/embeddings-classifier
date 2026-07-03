#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration.
This file is automatically discovered by pytest.
"""

import os
import pytest

from helpers import TestHelpers 

TEST_IMAGE_NAME = "crane-linear-model-runner:test"
USE_EXISTING_IMAGE_ENV = "DOCKER_TEST_USE_EXISTING_IMAGE"

@pytest.fixture(scope="session")
def docker_image():
    """
    Builds the Docker image once for the test session and yields its name.
    """
    use_existing_image = os.environ.get(USE_EXISTING_IMAGE_ENV, "").lower() in {"1", "true", "yes", "on"}
    built_in_fixture = False

    if use_existing_image:
        try:
            TestHelpers.sys_command(["docker", "image", "inspect", TEST_IMAGE_NAME])
            print(f"\nUsing existing Docker image: {TEST_IMAGE_NAME}")
        except RuntimeError:
            print(f"\nExisting Docker image not found, building: {TEST_IMAGE_NAME}")
            TestHelpers.sys_command(["docker", "build", "-t", TEST_IMAGE_NAME, "."])
            built_in_fixture = True
    else:
        print(f"\nBuilding Docker image: {TEST_IMAGE_NAME}")
        TestHelpers.sys_command(["docker", "build", "-t", TEST_IMAGE_NAME, "."])
        built_in_fixture = True
    
    # 'yield' passes the image name to the tests. The code below this
    # line will run after all tests in the session are complete.
    yield TEST_IMAGE_NAME
    
    if built_in_fixture:
        # Optional: Cleanup by removing the test image after the session
        print(f"\nCleaning up Docker image: {TEST_IMAGE_NAME}")
        TestHelpers.sys_command(["docker", "image", "rm", "-f", TEST_IMAGE_NAME])
