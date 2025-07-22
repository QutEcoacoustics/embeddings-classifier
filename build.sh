FULL_IMAGE_NAME="qutecoacoustics/linear-model-runner:latest"

docker buildx build \
    --platform linux/amd64 \
    --tag "$FULL_IMAGE_NAME" \
    --load \
    .