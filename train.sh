#!/usr/bin/env bash

rm -rf lightning_logs
python -m confidentaugmentation
python -m confidentaugmentation --selectively-backpropagate
python -m confidentaugmentation --selectively-backpropagate --mapie-alpha 0.05
python -m confidentaugmentation --selectively-backpropagate --mapie-alpha 0.03

