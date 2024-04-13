#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

IMAGE_SIZE=32

python -m citl train Cityscapes efficientnet-b0 \
    "--augmentation-policy-path=./policies/noop.yaml" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"

# python -m citl train CIFAR10 mnasnet_small --image-size=$IMAGE_SIZE \
#     "--augmentation-policy-path=./policies/cifar10.${IMAGE_SIZE}.yaml" \
#     "--selectively-backpropagate" \
#     "--mapie-alpha=0.10" \
#     "--lr-method=plateau" \
#     "--mapie-method=score"

# python -m citl train MNIST mobilenetv2_035 --image-size=$IMAGE_SIZE \
#     "--greyscale" \
#     "--augmentation-policy-path=./policies/mnist.yaml" \
#     "--selectively-backpropagate" \
#     "--mapie-alpha=0.10" \
#     "--lr-method=plateau" \
#     "--mapie-method=score"
