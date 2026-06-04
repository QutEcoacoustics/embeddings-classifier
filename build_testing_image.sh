#!/bin/bash

DOCKERHUB_USERNAME="qutecoacoustics"
IMAGE_NAME="crane-linear-model-runner"
IMAGE_VERSION="test"
DOCKERFILE_PATH="./Dockerfile"
BUILD_CONTEXT_PATH="." # This is typically '.' for the current directory

# --- IMPORTANT: Ensure Docker Buildx is set up ---
# The image is loaded locally via --load (single platform for the active builder/runtime).
# If you haven't already, create a builder (only needs to be done once per machine/setup)
# docker buildx create --name mybuilder --use
# Then, when you use it for the first time or after a restart:
# docker buildx inspect mybuilder --bootstrap

echo "--- Starting Docker Test Image Build Script (Local Load) ---"
echo ""

FULL_IMAGE_TAG="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_VERSION}"
echo "Building and tagging image as: ${FULL_IMAGE_TAG}"

echo "Building Docker image from ${DOCKERFILE_PATH} with context ${BUILD_CONTEXT_PATH}..."

docker buildx build \
    -f "${DOCKERFILE_PATH}" \
    -t "${FULL_IMAGE_TAG}" \
    --load \
    "${BUILD_CONTEXT_PATH}"  # This should be the final argument


# Check if the local build and load was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully built and loaded local image ${FULL_IMAGE_TAG}."
    echo "Use it locally with: docker run --rm ${FULL_IMAGE_TAG} version"
else
    echo ""
    echo "ERROR: Docker image build failed."
    exit 1
fi

echo ""
echo "--- Docker Test Image Build Script Finished ---"