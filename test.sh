#!/usr/bin/env bash

set -eux

python -m citl test CelebA resnet18 \
    "--alpha=0.01" \
    "--checkpoint=./epoch=21-val_accuracy=0.789.bin"

