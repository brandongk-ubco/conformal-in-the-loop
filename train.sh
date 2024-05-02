#!/usr/bin/env bash

set -eux

rm -rf lightning_logs/backprop_* || true
rm .*.ckpt || true

IMAGE_SIZE=224

python -m citl train CIFAR10 mnasnet_small --image-size=$IMAGE_SIZE \
    "--augmentation-policy-path=./policies/cifar10.${IMAGE_SIZE}.yaml" \
    "--selectively-backpropagate" \
    "--alpha=0.10" \
    "--lr-method=plateau" \
    "--method=score"

python -m citl train CIFAR10 mnasnet_small --image-size=$IMAGE_SIZE \
    "--augmentation-policy-path=./policies/cifar10.${IMAGE_SIZE}.yaml" \
    "--no-selectively-backpropagate" \
    "--alpha=0.10" \
    "--lr-method=plateau" \
    "--method=score"

python -m citl train CityscapesFine efficientnet-b0 \
    "--augmentation-policy-path=./policies/cityscapes.yaml" \
    "--selectively-backpropagate" \
    "--alpha=0.10" \
    "--lr-method=plateau" \
    "--method=score"

python -m citl train CityscapesFine efficientnet-b0 \
    "--augmentation-policy-path=./policies/cityscapes.yaml" \
    "--no-selectively-backpropagate" \
    "--alpha=0.10" \
    "--lr-method=plateau" \
    "--method=score"

# python -m citl standardtrain CIFAR10 mnasnet_small --image-size=$IMAGE_SIZE \
#     "--augmentation-policy-path=./policies/cifar10.${IMAGE_SIZE}.yaml" \
#     "--lr-method=plateau"

# python -m citl standardtrain Cityscapes efficientnet-b0 \
#     "--augmentation-policy-path=./policies/cityscapes.yaml" \
#     "--lr-method=plateau"
