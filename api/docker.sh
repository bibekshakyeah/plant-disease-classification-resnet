#!/bin/bash
docker build -t plant-disease-classification-resnet .
docker run -d -p 8088:8002 -v .:/app  cnn-api 