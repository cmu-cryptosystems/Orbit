#!/bin/bash

model=CompPart
act=SiLU

for no_comp in "" "--nocomp"; do
    for no_part in "" "--nopart"; do
        echo "Running Orbit optimization for 64k model: $model, activation: $act, no_comp: $no_comp, no_part: $no_part"
        python scripts/optimizer/orbit/run_orbit.py --model $model --act $act --n 64 --Lm 16 --Sw 40 $no_comp $no_part
        echo "  Finished Orbit optimization for model: $model, activation: $act, no_comp: $no_comp, no_part: $no_part"
    done
done
