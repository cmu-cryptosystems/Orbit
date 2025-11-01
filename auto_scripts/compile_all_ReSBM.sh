#!/bin/bash

acts=("ReLU" "SiLU")
model_16=("ResNet" "AlexNet" "SqueezeNet" "VGG16")
model_32=("ResNet" "AlexNet" "SqueezeNet" "VGG16" "MobileNet")
model_64=("ResNet" "AlexNet" "SqueezeNet" "VGG16" "MobileNet")

# echo "Start Default Running all 64k ReSBM"
# for model in "${model_64[@]}"; do
#     for act in "${acts[@]}"; do
#         echo "    Running model: $model, act: $act"
#         python -u -m scripts.optimizer.ReSBM.optimizer --inputfile mlirs_input/${model}${act}64k.mlir --outputfile mlirs_output/ReSBM_1_14_${model}${act}64k.mlir --costjson cost_models/lattigo_nepnew_64k_1_14_sim_vari.json > mlirs_output/ReSBM_1_14_${model}${act}64k.log 2>&1
#         echo "        Finished running model: $model, act: $act"
#     done
# done
# echo "Finished Default Running all 64k ReSBM"

echo "Start Default Running all 64k ReSBM"
for model in "${model_32[@]}"; do
    for act in "${acts[@]}"; do
        echo "    Running model: $model, act: $act"
        python -u -m scripts.optimizer.ReSBM.optimizer --inputfile mlirs_input/${model}${act}64k.mlir --outputfile mlirs_output/ReSBM/ReSBM_3_16_${model}${act}64k.mlir --costjson cost_models/profiled_LATTIGONEW_CPU64k_3_16_sim_vari.json > mlirs_output/ReSBM/ReSBM_3_16_${model}${act}64k.log 2>&1
        echo "        Finished running model: $model, act: $act"
    done
done
echo "Finished Default Running all 64k ReSBM"

# echo "Start Default Running all 16k ReSBM"
# for model in "${model_16[@]}"; do
#     for act in "${acts[@]}"; do
#         echo "    Running model: $model, act: $act"
#         python -u -m scripts.optimizer.ReSBM.optimizer --inputfile mlirs_input/${model}${act}16k.mlir --outputfile mlirs_output/ReSBM_1_14_${model}${act}16k.mlir --costjson cost_models/lattigo_nepnew_16k_1_14_sim_vari.json > mlirs_output/ReSBM_1_14_${model}${act}16k.log 2>&1
#         echo "        Finished running model: $model, act: $act"
#     done
# done
# echo "Finished Default Running all 16k ReSBM"