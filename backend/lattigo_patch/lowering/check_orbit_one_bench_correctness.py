import argparse
import os
import subprocess
import shutil

def get_result(file_path:str):
    results = []
    with open(file_path, 'r') as f:
        length = int(f.readline().strip())
        assert length >= 10
        for _ in range(10):
            line = f.readline().strip()
            results.append(float(line))
    res_max = None
    res_ans = None
    for class_idx in range(10):
        if res_max is None or results[class_idx] > res_max:
            res_max = results[class_idx]
            res_ans = class_idx
    return res_ans

def get_true_labels(file_path:str):
    true_labels = dict()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(':')
                input_file = parts[0].strip()
                assert input_file.startswith("input") and input_file.endswith(".txt")
                input_index = int(input_file[len("input"):-len(".txt")])
                label = int(parts[1].strip())
                true_labels[input_index] = label
    return true_labels

def get_inf_norm(vec1, vec2):
    assert len(vec1) == len(vec2)
    max_diff = 0.0
    # max_diff_val = None
    for i in range(len(vec1)):
        diff = abs(vec1[i] - vec2[i])
        if diff > max_diff:
            max_diff = diff
            # max_diff_val = (i, vec1[i], vec2[i])
    # print(f"  Max diff at index {max_diff_val[0]}: ct_val={max_diff_val[1]}, pt_val={max_diff_val[2]}, diff={max_diff}")
    return max_diff

def get_ptct_diff(ct_file:str, pt_file:str):
    ct_vals = []
    with open(ct_file, 'r') as f:
        length = int(f.readline().strip())
        assert length >= 10
        for _ in range(length):
            line = f.readline().strip()
            ct_vals.append(float(line))
    pt_vals = []
    with open(pt_file, 'r') as f:
        length = int(f.readline().strip())
        assert length >= 10
        for _ in range(length):
            line = f.readline().strip()
            pt_vals.append(float(line))
    return get_inf_norm(ct_vals, pt_vals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orbit ORBIT Checker')

    parser.add_argument('--model', type=str, required=True, choices=["ResNet", "AlexNet", "SqueezeNet", "MobileNet", "VGG16"], help='Model architecture')
    parser.add_argument('--act', type=str, required=True, choices=["ReLU", "SiLU"], help='Activation function')
    parser.add_argument('--n', type=int, required=True, choices=[16, 64], help='CKKS Vector Size')
    parser.add_argument('--Lm', type=int, default=16, help='Maximum Level Budget')
    parser.add_argument('--Sw', type=int, default=40, help='Waterline Scale')
    parser.add_argument('--runs', type=int, required=True, choices=range(1001), help='Number of samples to check')
    parser.add_argument('--mode', type=str, choices=['PyTorch-Orbit-ct', 'PyTorch-Orbit-pl', 'Orbit-ct-pl'], default='PyTorch-Orbit-ct', help='Which mode to check correctness')
    args = parser.parse_args()
    
    pl_ref_dir = f"../../../input_data/{args.n}k/{args.model.lower()}/{args.act.lower()}/plrefs"
    orbit_dir = f"../../../mlirs_execute/orbit/{args.n}/{args.model}/{args.act}"
    true_labels = get_true_labels(f"../../../input_data/{args.n}k/{args.model.lower()}/{args.act.lower()}/true_labels.txt")
    
    if "PyTorch" in args.mode:
        corr_ref = 0
        for i in range(args.runs):
            ref_label = get_result(f"{pl_ref_dir}/plref{i}.txt")
            true_label = true_labels[i]
            if ref_label == true_label:
                corr_ref += 1
        
        accuracy_ref = corr_ref / args.runs
        print(f"Checked {args.runs} PyTorch outputs. Accuracy: {accuracy_ref*100:.2f}%")
    
    if "-ct" in args.mode:
        corr_orbit_ct = 0
        for i in range(args.runs):
            orbit_file = f"{orbit_dir}/orbit_{args.model}{args.act}{args.n}k_Lm{args.Lm}_Sw{args.Sw}_bypass_noqbp_comp_part_run{i}.out"
            orbit_label = get_result(orbit_file)
            true_label = true_labels[i]
            if orbit_label == true_label:
                corr_orbit_ct += 1
            
        accuracy_orbit_ct = corr_orbit_ct / args.runs
        print(f"Checked {args.runs} Orbit (ct) outputs. Accuracy: {accuracy_orbit_ct*100:.2f}%")
    
    if "-pl" in args.mode:
        corr_orbit_pl = 0
        for i in range(args.runs):
            orbit_file = f"{orbit_dir}/orbit_{args.model}{args.act}{args.n}k_Lm{args.Lm}_Sw{args.Sw}_bypass_noqbp_comp_part_run{i}_pl.out"
            orbit_label = get_result(orbit_file)
            true_label = true_labels[i]
            if orbit_label == true_label:
                corr_orbit_pl += 1
            
        accuracy_orbit_pl = corr_orbit_pl / args.runs
        print(f"Checked {args.runs} Orbit (pl) outputs. Accuracy: {accuracy_orbit_pl*100:.2f}%")
    
    if args.mode == "Orbit-ct-pl":
        max_diff = 0.0
        for i in range(args.runs):
            orbit_pt_file = f"{orbit_dir}/orbit_{args.model}{args.act}{args.n}k_Lm{args.Lm}_Sw{args.Sw}_bypass_noqbp_comp_part_run{i}_pl.out"
            orbit_ct_file = f"{orbit_dir}/orbit_{args.model}{args.act}{args.n}k_Lm{args.Lm}_Sw{args.Sw}_bypass_noqbp_comp_part_run{i}.out"
            diff = get_ptct_diff(orbit_ct_file, orbit_pt_file)
            if diff > max_diff:
                max_diff = diff
            print(f"  Run {i} Diff: {diff:.8f}")
        print(f"Orbit max PT-CT diff over {args.runs} runs: {max_diff:.8f}")
    