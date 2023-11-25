#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

for augmentation_policy in noop cifar10; do
    for model_name in efficientnet-b0 efficientnet-b3; do
        for selectively_backpropagate in "--no-selectively-backpropagate" "--selectively-backpropagate"; do
            for mapie_alpha in 0.10 0.05 0.03; do
                python -m confidentaugmentation cifar10 \
                    "--model-name=${model_name}" \
                    "--augmentation-policy-path=./policies/${augmentation_policy}.yaml" \
                    "${selectively_backpropagate}" \
                    --mapie-alpha "${mapie_alpha}"
            done
        done
    done
done
