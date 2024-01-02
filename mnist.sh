#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

python -m confidentaugmentation train mnist \
    "--model-name=MicroNet" \
    "--augmentation-policy-path=./policies/mnist.yaml" \
    "--no-pretrained" \
    "--selectively-backpropagate" \
    "--mapie-alpha=0.10" \
    "--lr-method=plateau" \
    "--optimizer=Adam" \
    "--no-control-weight-decay" \
    "--no-control-pixel-dropout" \
    "--mapie-method=score"
