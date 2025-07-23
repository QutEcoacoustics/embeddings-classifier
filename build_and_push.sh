#!/bin/bash

DOCKERHUB_USERNAME="qutecoacoustics"
IMAGE_NAME="crane-linear-model-runner"
IMAGE_VERSION="1.0.0"
DOCKERFILE_PATH="./Dockerfile"
BUILD_CONTEXT_PATH="." # This is typically '.' for the current directory

# --- IMPORTANT: Ensure Docker Buildx is set up for multi-platform builds ---
# You need a 'builder' that supports multiple platforms.
# If you haven't already, create a builder (only needs to be done once per machine/setup)
# docker buildx create --name mybuilder --use
# Then, when you use it for the first time or after a restart:
# docker buildx inspect mybuilder --bootstrap

echo "--- Starting Docker Build and Push Script (Multi-Platform) ---"
echo ""

FULL_IMAGE_TAG="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_VERSION}"
echo "Building and tagging image as: ${FULL_IMAGE_TAG}"

echo "Building Docker image for multiple platforms (linux/amd64, linux/arm64) from ${DOCKERFILE_PATH} with context ${BUILD_CONTEXT_PATH}..."

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -f "${DOCKERFILE_PATH}" \
    -t "${FULL_IMAGE_TAG}" \
    --push \
    "${BUILD_CONTEXT_PATH}"  # This should be the final argument


# Check if the build and push was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully built and pushed multi-platform image ${FULL_IMAGE_TAG} to Docker Hub."
    echo "This image now supports both linux/amd64 (for your WSL) and linux/arm64 (for your Mac)."
else
    echo ""
    echo "ERROR: Docker image build or push failed for multi-platform image."
    exit 1
fi

echo ""
echo "--- Docker Build and Push Script Finished ---"