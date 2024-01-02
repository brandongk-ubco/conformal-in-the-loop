#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

dataset="cifar10"

for augmentation_policy in "cifar10.32"; do
    for model_name in "efficientnet_b0"; do
        for mapie_alpha in 0.10; do
            for selectively_backpropagate in "--selectively-backpropagate"; do
                for pretrained in "--pretrained"; do
                    for lr_method in "plateau"; do
                        for optimizer in "Adam"; do
                                    for mapie_method in "score"; do
                                        python -m confidentaugmentation train ${dataset} \
                                            "--model-name=${model_name}" \
                                            "--augmentation-policy-path=./policies/${augmentation_policy}.yaml" \
                                            "${pretrained}" \
                                            "${selectively_backpropagate}" \
                                            "--mapie-alpha=${mapie_alpha}" \
                                            "--lr-method=${lr_method}" \
                                            "--optimizer=${optimizer}" \
                                            "--mapie-method=${mapie_method}"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
