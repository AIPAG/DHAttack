#!/bin/bash
set -e
DATA_DIR="./data"
EXTRACT_DIR="${DATA_DIR}/cifar-10-batches-py"
TARGET_DIR="${DATA_DIR}/cifar-10-download"
TAR_PATH="${DATA_DIR}/cifar-10-python.tar.gz"

mkdir -p "${DATA_DIR}"
mkdir -p "${TARGET_DIR}"

wget -q --show-progress -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P "${DATA_DIR}"
tar -xvzf "${TAR_PATH}" -C "${DATA_DIR}"
mv "${EXTRACT_DIR}/"* "${TARGET_DIR}/"
rm -rf "${EXTRACT_DIR}"
rm -f "${TAR_PATH}"

echo "success!"