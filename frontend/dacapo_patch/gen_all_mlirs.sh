#!/bin/bash
shopt -s expand_aliases
source config.sh

echo "=== DEBUG INFO ==="
echo "HECATE variable: $HECATE"
echo "Checking if functions are defined:"
type hc-back-opt 2>/dev/null && echo "hc-back-opt function: OK" || echo "hc-back-opt function: NOT FOUND"
type hbt 2>/dev/null && echo "hbt alias: OK" || echo "hbt alias: NOT FOUND"
type hc-trace 2>/dev/null && echo "hc-trace alias: OK" || echo "hc-trace alias: NOT FOUND"
echo "=================="

declare -a Models=(
    "ResNet"
    "AlexNet"
    "VGG16"
    "MobileNet"
    "SqueezeNet"
)

declare -a Acts=(
    "ReLU16k"
    "SiLU16k"
    "ReLU64k"
    "SiLU64k"
)

num_models=${#Models[@]}
num_acts=${#Acts[@]}


mkdir -p ../log
mkdir -p ../../mlirs_input
mkdir -p ../../input_constants
for (( i=0; i<num_models; i++ ))
do
    for (( j=0; j<num_acts; j++ ))
    do
        read model act <<< "${Models[i]} ${Acts[j]}"

        # if model is MobileNet and act is SiLU16k or ReLU16k, skip
        if [[ "$model" == "MobileNet" && ("$act" == "SiLU16k" || "$act" == "ReLU16k") ]]; then
            echo "Skipping task: ${model}${act}"
            continue
        fi
        echo "Running task: ${model}${act}"

        ## Generate MLIR
        hc-trace ${model}${act} > ../log/${model}${act}_trace.txt 2>../log/${model}${act}_trace.err
        python python/propagate_comments.py examples/traced/${model}${act}.mlir ../../mlirs_input/${model}${act}.mlir
        cp ../../mlirs_input/${model}${act}.mlir examples/traced/${model}${act}.mlir
        cp examples/traced/_hecate_${model}${act}.cst ../../input_constants/${model}${act}_hecate.cst

        echo "End running task: ${model}${act}"
    done
done