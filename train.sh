#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

python -m citl train MNIST mobilenetv2_035 32 \
    "--greyscale" \
    "--augmentation-policy-path=./policies/mnist.yaml" \
    "--selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"
