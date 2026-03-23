#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

if [ ! -d "pretrained" ]; then
    mkdir pretrained
fi

if [ ! -d "pretrained/sam2" ]; then
    mkdir pretrained/sam2
fi

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url -O pretrained/sam2/sam2.1_hiera_large.pt || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

if [ ! -d "pretrained/WiLoR" ]; then
    mkdir pretrained/WiLoR
fi

echo "Downloading detector.pt and wilor_final.ckpt..."
$CMD https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -O pretrained/WiLoR/detector.pt || { echo "Failed to download detector.pt"; exit 1; }
$CMD https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -O pretrained/WiLoR/wilor_final.ckpt || { echo "Failed to download wilor_final.ckpt"; exit 1; }
$CMD https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/model_config.yaml -O pretrained/WiLoR/model_config.yaml || { echo "Failed to download model_config.yaml"; exit 1; }

echo "All checkpoints are downloaded successfully."