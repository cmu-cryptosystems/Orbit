#!/bin/bash

for model in AlexNet SqueezeNet VGG16 ResNet; do
    for act in SiLU ReLU; do
        echo "Running Orbit optimization for 16k model: $model, activation: $act"
        python3 scripts/optimizer/orbit/run_orbit.py --threads "${ORBIT_THREADS:-32}" --model $model --act $act --n 16 --Lm 16 --Sw 40
        echo "  Finished Orbit optimization for model: $model, activation: $act"
    done
done