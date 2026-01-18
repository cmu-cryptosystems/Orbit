#!/bin/bash

for model in ResNet; do
    for act in ReLU SiLU; do
        echo "Running Orbit optimization for 64k (no bypass) model: $model, activation: $act"
        python3 scripts/optimizer/orbit/run_orbit.py --model $model --act $act --n 64 --Lm 16 --Sw 40 --nobypass
        echo "  Finished Orbit optimization for model: $model, activation: $act"
    done
done

for model in ResNet; do
    for act in SiLU ReLU; do
        echo "Running Orbit optimization for 16k (no bypass) model: $model, activation: $act"
        python3 scripts/optimizer/orbit/run_orbit.py --model $model --act $act --n 16 --Lm 16 --Sw 40 --nobypass
        echo "  Finished Orbit optimization for model: $model, activation: $act"
    done
done