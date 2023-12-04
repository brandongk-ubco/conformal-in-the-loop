#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

for augmentation_policy in cifar10; do
    for model_name in efficientnet_b0 efficientnet_b1 efficientnet_b3 efficientnet_b5; do
        for mapie_alpha in 0.10; do
            for selectively_backpropagate in "--selectively-backpropagate" "--no-selectively-backpropagate"; do
                for pretrained in "--pretrained"; do
                    for use_pid in "--use-pid"; do
                        python -m confidentaugmentation train cifar10 \
                            "--model-name=${model_name}" \
                            "--augmentation-policy-path=./policies/${augmentation_policy}.yaml" \
                            "${pretrained}" \
                            "${selectively_backpropagate}" \
                            "${use_pid}" \
                            "--mapie-alpha=${mapie_alpha}"
                    done
                done
            done
        done
    done
done
