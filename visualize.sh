#!/usr/bin/env bash

set -eux

rm -rf visualizations

for augmentation_policy in cifar10 noop; do
    python -m confidentaugmentation visualize cifar10 \
        "--augmentation-policy-path=./policies/${augmentation_policy}.yaml"
done
