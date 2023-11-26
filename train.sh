#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

python -m confidentaugmentation train cifar10 \
    "--model-name=ViT" \
    "--augmentation-policy-path=./policies/cifar10.yaml" \
    "--selectively-backpropagate" \
    "--mapie-alpha=0.10"

python -m confidentaugmentation train cifar10 \
    "--model-name=ViT" \
    "--augmentation-policy-path=./policies/cifar10.yaml" \
    "--mapie-alpha=0.10"

exit 0

for augmentation_policy in noop cifar10; do
    for model_name in efficientnet-b0 efficientnet-b3; do
        for selectively_backpropagate in "--no-selectively-backpropagate" "--selectively-backpropagate"; do
            for mapie_alpha in 0.10 0.05 0.03; do
                for pretrained in "--pretrained" "--no-pretrained"; do
                    python -m confidentaugmentation train cifar10 \
                        "--model-name=${model_name}" \
                        "--augmentation-policy-path=./policies/${augmentation_policy}.yaml" \
                        "${pretrained}" \
                        "${selectively_backpropagate}" \
                        "--mapie-alpha=${mapie_alpha}"
                done
            done
        done
    done
done
