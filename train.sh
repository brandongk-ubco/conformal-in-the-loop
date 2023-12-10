#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

for augmentation_policy in "cifar10"; do
    for model_name in "efficientnet_b0"; do
        for mapie_alpha in 0.10; do
            for selectively_backpropagate in "--selectively-backpropagate"; do
                for pretrained in "--pretrained"; do
                    for lr_method in "plateau" "uncertainty"; do
                        for optimizer in "Adam"; do
                            for control_weight_decay in "--no-control-weight-decay" "--control-weight-decay"; do
                                for control_pixel_dropout in "--no-control-pixel-dropout" "--control-pixel-dropout" ; do
                                    python -m confidentaugmentation train cifar10 \
                                        "--model-name=${model_name}" \
                                        "--augmentation-policy-path=./policies/${augmentation_policy}.yaml" \
                                        "${pretrained}" \
                                        "${selectively_backpropagate}" \
                                        "--mapie-alpha=${mapie_alpha}" \
                                        "--lr-method=${lr_method}" \
                                        "--optimizer=${optimizer}" \
                                        "${control_weight_decay}" \
                                        "${control_pixel_dropout}"
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
