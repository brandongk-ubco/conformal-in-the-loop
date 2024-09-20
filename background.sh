#!/usr/bin/env bash

export $(cat .env | xargs)

nohup ./train.sh > output.log 2>&1 &