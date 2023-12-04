#!/usr/bin/env bash

set -eux

mkdir -p datasets/imagenet

cd datasets/imagenet
at-get a306397ccf9c2ead27155983c254227c0fd938e2

mkdir -p train && tar -xvf ILSVRC2012_img_train.tar --directory train
(cd train && ls *.tar |  sed -e 's/\.tar$//' | parallel --progress -I@ -j4 'mkdir -p @; tar -xf @.tar --directory @; rm @.tar')


wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar