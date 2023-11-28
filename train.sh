#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

for augmentation_policy in cifar10 noop; do
    for model_name in efficientnet_b0 efficientnet_b3; do
        for mapie_alpha in 0.10 0.05 0.03; do
            for selectively_backpropagate in "--no-selectively-backpropagate" "--selectively-backpropagate"; do
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
