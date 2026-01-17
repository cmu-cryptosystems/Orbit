#!/bin/bash

for model in AlexNet MobileNet SqueezeNet VGG16 ResNet; do
    for act in SiLU ReLU; do
        echo "Running Orbit optimization for 64k sim-vari-bts model: $model, activation: $act"
        python scripts/optimizer/orbit/run_orbit.py --model $model --act $act --n 64 --Lm 16 --Sw 40 --sim-vari
        echo "  Finished Orbit optimization for model: $model, activation: $act"
    done
done