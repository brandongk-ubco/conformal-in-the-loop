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

# levels=(0.3 0.4 0.5)

# for level in "${levels[@]}"
# do

#     # python -m citl standardtrain CIFAR10UB mnasnet_small \
#     #     "--augmentation-policy-path=./policies/cifar10.yaml" \
#     #     "--noise-level=${level}" \
#     #     "--loss-function=cross_entropy" \
#     #     "--lr-method=plateau"

#     python -m citl standardtrain CIFAR10UB mnasnet_small \
#         "--augmentation-policy-path=./policies/cifar10.yaml" \
#         "--noise-level=${level}" \
#         "--loss-function=focal" \
#         "--lr-method=plateau"
# done

# python -m citl standardtrain CityscapesFine efficientnet-b0 \
#     "--augmentation-policy-path=./policies/cityscapes.yaml" \
#     "--lr-method=plateau"

# # NORMAL BACKPROP BASELINES

# python -m citl train CelebA resnet18 \
#     "--no-selectively-backpropagate" \
#     "--alpha=0.10" \
#     "--augmentation-policy-path=./policies/celeba.yaml" \
#     "--lr-method=plateau"

# python -m citl train CIFAR10UB mnasnet_small \
#     "--augmentation-policy-path=./policies/cifar10.yaml" \
#     "--no-selectively-backpropagate" \
#     "--alpha=0.10" \
#     "--lr-method=plateau" \
#     "--method=score"

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

numbers=(0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19)
level=0.2

for alpha in "${numbers[@]}"
do

    python -m citl train CIFAR10UB mnasnet_small \
        "--augmentation-policy-path=./policies/cifar10.yaml" \
        "--selectively-backpropagate" \
        "--alpha=${alpha}" \
        "--noise-level=${level}" \
        "--loss-function=cross_entropy" \
        "--lr-method=plateau"

    # python -m citl train CelebA resnet18 \
    #     "--selectively-backpropagate" \
    #     "--alpha=${alpha}" \
    #     "--augmentation-policy-path=./policies/celeba.yaml" \
    #     "--lr-method=plateau"

    # python -m citl train DFire mnasnet_small \
    #     "--augmentation-policy-path=./policies/DFire.yaml" \
    #     "--selectively-backpropagate" \
    #     "--alpha=${alpha}" \
    #     "--lr-method=plateau" \
    #     "--method=score"

    # python -m citl train CIFAR10 mnasnet_small  \
    #     "--augmentation-policy-path=./policies/cifar10.yaml" \
    #     "--selectively-backpropagate" \
    #     "--alpha=${alpha}" \
    #     "--lr-method=plateau" \
    #     "--method=score"

    # python -m citl train CityscapesFine efficientnet-b0 \
    #     "--augmentation-policy-path=./policies/cityscapes.yaml" \
    #     "--selectively-backpropagate" \
    #     "--alpha=${alpha}" \
    #     "--lr-method=plateau" \
    #     "--method=score"
done


