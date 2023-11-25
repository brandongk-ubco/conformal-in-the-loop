#!/usr/bin/env bash

set -eux

rm -rf lightning_logs

# python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/noop.yaml
# python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate
# python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate --mapie-alpha 0.05
# python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate --mapie-alpha 0.03

# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/noop.yaml
# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate
# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate --mapie-alpha 0.05
# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate --mapie-alpha 0.03


# python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/cifar10.yaml
python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/noop.yaml
python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/noop.yaml --pretrained
python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/cifar10.yaml --selectively-backpropagate
python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/cifar10.yaml --selectively-backpropagate --pretrained
# python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/cifar10.yaml --selectively-backpropagate --mapie-alpha 0.05
# python -m confidentaugmentation cifar10 --augmentation-policy-path=./policies/cifar10.yaml --selectively-backpropagate --mapie-alpha 0.03

# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/cifar10.yaml
# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/cifar10.yaml --selectively-backpropagate
# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/cifar10.yaml --selectively-backpropagate --mapie-alpha 0.05
# python -m confidentaugmentation cifar10 --model-name=efficientnet-b3 --augmentation-policy-path=./policies/cifar10.yaml --selectively-backpropagate --mapie-alpha 0.03
