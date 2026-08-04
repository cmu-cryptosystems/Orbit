#!/bin/bash

rm -rf qbps_cache_L16_LB3_Sf51_Sw40_16k/
for model in AlexNet SqueezeNet VGG16 ResNet; do
    for act in SiLU ReLU; do
        echo "Running Orbit optimization for 16k (reusing qbp) model: $model, activation: $act"
        python3 scripts/optimizer/orbit/run_orbit.py --threads "${ORBIT_THREADS:-32}" --model $model --act $act --n 16 --Lm 16 --Sw 40 --qbp
        echo "  Finished Orbit optimization for model: $model, activation: $act"
    done
done
