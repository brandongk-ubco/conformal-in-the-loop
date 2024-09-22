#!/usr/bin/env bash

set -eux

python -m citl test CityscapesFine efficientnet-b0 \
    "--alpha=0.10" \
    "--quantile=0.9194809794425964" \
    "--checkpoint=./epoch=154-val_accuracy=0.765.bin"

