#!/bin/bash

for model in AlexNet MobileNet ResNet SqueezeNet VGG16; do
    for act in ReLU SiLU; do
        echo "Running Saturn optimization for 64k model: $model, activation: $act"
        python scripts/optimizer/saturn/run_saturn.py --model $model --act $act --n 64 --Lm 16 --Sw 40
        echo "  Finished Saturn optimization for model: $model, activation: $act"
    done
done

for model in AlexNet ResNet SqueezeNet VGG16; do
    for act in ReLU SiLU; do
        echo "Running Saturn optimization for 16k model: $model, activation: $act"
        python scripts/optimizer/saturn/run_saturn.py --model $model --act $act --n 16 --Lm 16 --Sw 40
        echo "  Finished Saturn optimization for model: $model, activation: $act"
    done
done