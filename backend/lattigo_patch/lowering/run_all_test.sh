#!/bin/bash

acts=("Silu" "Relu")

lm=16
Sw=40

models16=("AlexNet" "ResNet" "SqueezeNet" "VGG16" )
models64=("AlexNet" "ResNet" "SqueezeNet" "VGG16" "MobileNet")

n=16
for run in {0..2}; do
    for model in "${models16[@]}"; do
        for act in "${acts[@]}"; do
            echo "Running 16k model: $model, act: $act, run: $run"
            python run_one_test_tmp.py --model $model --act $act --n $n --Lm $lm --Sw $Sw --cmt t16_nobypass_noqbp --run $run --lcmt _new_rot_upscale
            echo "  Finished running model: $model, act: $act, run: $run"
        done
    done
done

# n=64
# for model in "${models64[@]}"; do
#     for act in "${acts[@]}"; do
#         echo "Running 64k model: $model, act: $act"
#         python run_one_test_tmp.py --model $model --act $act --n $n --Lm $lm --Sw $Sw --cmt t16_nobypass_noqbp --run 0 --lcmt _silu_first
#         echo "  Finished running model: $model, act: $act"
#     done
# done

# acts=("Relu" "Silu")
# n=16
# for model in "${models16[@]}"; do
#     for act in "${acts[@]}"; do
#         echo "Running 16k model: $model, act: $act"
#         python run_one_test_tmp.py --model $model --act $act --n $n --Lm $lm --Sw $Sw --cmt t16_nobypass_noqbp --run 0 --lcmt _relu_first
#         echo "  Finished running model: $model, act: $act"
#     done
# done

# # n=64
# # for model in "${models64[@]}"; do
# #     for act in "${acts[@]}"; do
# #         echo "Running 64k model: $model, act: $act"
# #         python run_one_test_tmp.py --model $model --act $act --n $n --Lm $lm --Sw $Sw --cmt t16_nobypass_noqbp --run 0 --lcmt _relu_first
# #         echo "  Finished running model: $model, act: $act"
# #     done
# # done