#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

for augmentation_policy in "cifar10"; do
    for model_name in "efficientnet_b0" "efficientnet_b1" "efficientnet_b3" "efficientnet_b5"; do
        for mapie_alpha in 0.10; do
            for selectively_backpropagate in "--selectively-backpropagate" "--no-selectively-backpropagate"; do
                for pretrained in "--pretrained" "--no-pretrained"; do
                    for use_pid in "--use-pid" "--no-use-pid"; do
                        for lr_method in "one_cycle" "plateau" "uncertainty"; do
                            for optimizer in "Adam"; do
                                for control_weight_decay in "--control-weight-decay" "--no-control-weight-decay"; do
                                    for control_pixel_dropout in "--control-pixel-dropout" "--no-control-pixel-dropout"; do
                                        python -m confidentaugmentation train cifar10 \
                                            "--model-name=${model_name}" \
                                            "--augmentation-policy-path=./policies/${augmentation_policy}.yaml" \
                                            "${pretrained}" \
                                            "${selectively_backpropagate}" \
                                            "${use_pid}" \
                                            "--mapie-alpha=${mapie_alpha}" \
                                            "--lr-method=${lr_method}" \
                                            "--optimizer=${optimizer}" \
                                            "${control_weight_decay}" \
                                            "${control_pixel_dropout}" 2>&1 | tee run.log
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
