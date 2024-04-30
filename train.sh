#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

IMAGE_SIZE=224

python -m citl train CIFAR10 mnasnet_small --image-size=$IMAGE_SIZE \
    "--augmentation-policy-path=./policies/cifar10.${IMAGE_SIZE}.yaml" \
    "--selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"

python -m citl train CIFAR10 mnasnet_small --image-size=$IMAGE_SIZE \
    "--augmentation-policy-path=./policies/cifar10.${IMAGE_SIZE}.yaml" \
    "--no-selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"

python -m citl train Cityscapes efficientnet-b0 \
    "--augmentation-policy-path=./policies/cityscapes.yaml" \
    "--selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"

python -m citl train Cityscapes efficientnet-b0 \
    "--augmentation-policy-path=./policies/cityscapes.yaml" \
    "--no-selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--mapie-method=score"

# python -m citl train MNIST mnasnet_small --image-size=$IMAGE_SIZE \
#     "--greyscale" \
#     "--augmentation-policy-path=./policies/mnist.yaml" \
#     "--selectively-backpropagate" \
#     "--mapie-alpha=0.10" \
#     "--lr-method=plateau" \
#     "--mapie-method=score"
