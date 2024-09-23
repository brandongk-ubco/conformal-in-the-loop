#!/usr/bin/env bash

set -eux

python -m citl test CityscapesFine efficientnet-b0 \
    "--alpha=0.01" \
    "--checkpoint=./last.bin"

