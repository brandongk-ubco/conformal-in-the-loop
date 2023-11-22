#!/usr/bin/env bash

rm -rf lightning_logs
# python -m confidentaugmentation --augmentation-policy-path=./policies/noop.yaml
# python -m confidentaugmentation --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate
# python -m confidentaugmentation --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate --mapie-alpha 0.05
# python -m confidentaugmentation --augmentation-policy-path=./policies/noop.yaml --selectively-backpropagate --mapie-alpha 0.03

python -m confidentaugmentation --augmentation-policy-path=./policies/flip.yaml
python -m confidentaugmentation --augmentation-policy-path=./policies/flip.yaml --selectively-backpropagate
python -m confidentaugmentation --augmentation-policy-path=./policies/flip.yaml --selectively-backpropagate --mapie-alpha 0.05
python -m confidentaugmentation --augmentation-policy-path=./policies/flip.yaml --selectively-backpropagate --mapie-alpha 0.03
