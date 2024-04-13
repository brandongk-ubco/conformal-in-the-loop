#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

IMAGE_SIZE=32

python -m citl train Cityscapes mobilenetv2_035 $IMAGE_SIZE \
    "--augmentation-policy-path=./policies/noop.yaml" \
    "--selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"

# python -m citl train CIFAR10 mobilenetv2_035 $IMAGE_SIZE \
#     "--augmentation-policy-path=./policies/cifar10.${IMAGE_SIZE}.yaml" \
#     "--selectively-backpropagate" \
#     "--mapie-alpha=0.10" \
#     "--lr-method=plateau" \
#     "--mapie-method=score"

# python -m citl train MNIST mobilenetv2_035 32 \
#     "--greyscale" \
#     "--augmentation-policy-path=./policies/mnist.yaml" \
#     "--selectively-backpropagate" \
#     "--mapie-alpha=0.10" \
#     "--lr-method=plateau" \
#     "--mapie-method=score"
