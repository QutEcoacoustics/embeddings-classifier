#!/bin/bash

DOCKERHUB_USERNAME="qutecoacoustics"  
IMAGE_NAME="crane-linear-model-runner"                        
IMAGE_VERSION="1.0.0"
DOCKERFILE_PATH="./Dockerfile"
BUILD_CONTEXT_PATH="."


echo "--- Starting Docker Build and Push Script ---"
echo ""

FULL_IMAGE_TAG="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_VERSION}"
echo "Building and tagging image as: ${FULL_IMAGE_TAG}"


echo "Building Docker image from ${DOCKERFILE_PATH} with context ${BUILD_CONTEXT_PATH}..."
docker buildx build --load -f "${DOCKERFILE_PATH}" -t "${FULL_IMAGE_TAG}" "${BUILD_CONTEXT_PATH}"

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
    echo ""

    echo "Pushing image ${FULL_IMAGE_TAG} to Docker Hub..."
    docker push "${FULL_IMAGE_TAG}"

    # Check if the push was successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "Successfully pushed ${FULL_IMAGE_TAG} to Docker Hub."
    else
        echo ""
        echo "ERROR: Failed to push Docker image to Docker Hub."
        exit 1 
    fi
else
    echo ""
    echo "ERROR: Docker image build failed."
    exit 1 
fi

echo ""
echo "--- Docker Build and Push Script Finished ---"