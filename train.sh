#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

# python -m citl train CelebA resnet18 \
#     "--selectively-backpropagate" \
#     "--augmentation-policy-path=./policies/celeba.yaml" \
#     "--lr-method=plateau"

# STANDARD TRAINING BASELINES

# python -m citl standardtrain CelebA resnet18 \
#     "--augmentation-policy-path=./policies/celeba.yaml" \
#     "--lr-method=plateau"

# python -m citl standardtrain DFire mnasnet_small \
#     "--augmentation-policy-path=./policies/DFire.yaml" \
#     "--lr-method=plateau"

# python -m citl standardtrain CIFAR10UB mnasnet_small \
#     "--augmentation-policy-path=./policies/cifar10.yaml" \
#     "--lr-method=plateau"

# python -m citl standardtrain CityscapesFine efficientnet-b0 \
#     "--augmentation-policy-path=./policies/cityscapes.yaml" \
#     "--lr-method=plateau"

# # NORMAL BACKPROP BASELINES

# python -m citl train CelebA resnet18 \
#     "--no-selectively-backpropagate" \
#     "--alpha=0.10" \
#     "--augmentation-policy-path=./policies/celeba.yaml" \
#     "--lr-method=plateau"

python -m citl train CIFAR10UB mnasnet_small \
    "--augmentation-policy-path=./policies/cifar10.yaml" \
    "--no-selectively-backpropagate" \
    "--alpha=0.10" \
    "--lr-method=plateau" \
    "--method=score"

# python -m citl train DFire mnasnet_small \
#     "--augmentation-policy-path=./policies/DFire.yaml" \
#     "--no-selectively-backpropagate" \
#     "--alpha=0.10" \
#     "--lr-method=plateau" \
#     "--method=score"

# python -m citl train CityscapesFine efficientnet-b0 \
#     "--augmentation-policy-path=./policies/cityscapes.yaml" \
#     "--no-selectively-backpropagate" \
#     "--alpha=0.10" \
#     "--lr-method=plateau" \
#     "--method=score"

# # METHOD ALPHA SWEEP

# numbers=(0.01)

# for alpha in "${numbers[@]}"
# do

#     python -m citl train CIFAR10UB mnasnet_small \
#         "--augmentation-policy-path=./policies/cifar10.yaml" \
#         "--selectively-backpropagate" \
#         "--alpha=${alpha}" \
#         "--lr-method=plateau"

#     # python -m citl train CelebA resnet18 \
#     #     "--selectively-backpropagate" \
#     #     "--alpha=${alpha}" \
#     #     "--augmentation-policy-path=./policies/celeba.yaml" \
#     #     "--lr-method=plateau"

#     # python -m citl train DFire mnasnet_small \
#     #     "--augmentation-policy-path=./policies/DFire.yaml" \
#     #     "--selectively-backpropagate" \
#     #     "--alpha=${alpha}" \
#     #     "--lr-method=plateau" \
#     #     "--method=score"

#     # python -m citl train CIFAR10 mnasnet_small  \
#     #     "--augmentation-policy-path=./policies/cifar10.yaml" \
#     #     "--selectively-backpropagate" \
#     #     "--alpha=${alpha}" \
#     #     "--lr-method=plateau" \
#     #     "--method=score"

#     # python -m citl train CityscapesFine efficientnet-b0 \
#     #     "--augmentation-policy-path=./policies/cityscapes.yaml" \
#     #     "--selectively-backpropagate" \
#     #     "--alpha=${alpha}" \
#     #     "--lr-method=plateau" \
#     #     "--method=score"
# done


