#!/bin/bash
# docker build -t plant-disease-classification-resnet .
docker run --shm-size=6g --gpus all -it --rm -p 8888:8888 -v /home/bibek/avo-workspace/rnd/plant-disease-classification-resnet:/app  plant-disease-classification-resnet