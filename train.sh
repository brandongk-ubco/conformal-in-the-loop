#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

python -m citl train CIFAR10 efficientnet_b0 224 \
    "--augmentation-policy-path=./policies/cifar10.224.yaml" \
    "--selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"

# python -m citl train MNIST mobilenetv2_035 32 \
#     "--greyscale" \
#     "--augmentation-policy-path=./policies/mnist.yaml" \
#     "--selectively-backpropagate" \
#     "--mapie-alpha=0.10" \
#     "--lr-method=plateau" \
#     "--mapie-method=score"
