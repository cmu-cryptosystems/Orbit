#!/bin/bash

for model in AlexNet SqueezeNet VGG16 ResNet; do
    for act in ReLU SiLU; do
        echo "Running Saturn optimization for 16k model reusing qbp: $model, activation: $act"
        python scripts/optimizer/saturn/run_saturn.py --model $model --act $act --n 16 --Lm 16 --Sw 40 --qbp
        echo "  Finished Saturn optimization for model: $model, activation: $act"
    done
done