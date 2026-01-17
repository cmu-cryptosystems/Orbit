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
    parser = argparse.ArgumentParser(description='Orbit ILP Checker')

    parser.add_argument('--model', type=str, required=True, choices=["ResNet", "AlexNet", "SqueezeNet", "MobileNet", "VGG16"], help='Model architecture')
    parser.add_argument('--act', type=str, required=True, choices=["ReLU", "SiLU"], help='Activation function')
    parser.add_argument('--n', type=int, required=True, choices=[16, 64], help='CKKS Vector Size')
    parser.add_argument('--Lm', type=int, default=16, help='Maximum Level Budget')
    parser.add_argument('--Sw', type=int, default=40, help='Waterline Scale')
    parser.add_argument('--run', type=int, required=True, choices=range(1001), help='How many output files to check')
    args = parser.parse_args()
    
    
    pl_ref_dir = f"inputs/{args.n}k/{args.model.lower()}/{args.act.lower()}/plrefs"
    ilp_dir = f"execution_res/{args.n}/{args.model}/{args.act}"
    true_labels = get_true_labels(f"inputs/{args.n}k/{args.model.lower()}/{args.act.lower()}/true_labels.txt")
    
    corr_ref = 0
    for i in range(args.run):
        ref_label = get_result(f"{pl_ref_dir}/plref{i}.txt")
        true_label = true_labels[i]
        if ref_label == true_label:
            corr_ref += 1
    
    accuracy_ref = corr_ref / args.run
    print(f"Checked {args.run} ref outputs. Accuracy: {accuracy_ref*100:.2f}%")
    
    # corr_pl = 0
    # for i in range(args.run):
    #     pl_file = f"{ilp_dir}/{args.model}{args.act}{args.n}k_run{i}_pl.out"
    #     pl_label = get_result(pl_file)
    #     true_label = true_labels[i]
    #     if pl_label == true_label:
    #         corr_pl += 1
            
    # accuracy_pl = corr_pl / args.run
    # print(f"Checked {args.run} PL outputs. Accuracy: {accuracy_pl*100:.2f}%")
    
    corr_ilp = 0
    for i in range(args.run):
        ilp_file = f"{ilp_dir}/orbit_{args.model}{args.act}{args.n}k_Lm{args.Lm}_Sw{args.Sw}_Csw15_bypass_noqbp_comp_part_run{i}.out"
        ilp_label = get_result(ilp_file)
        true_label = true_labels[i]
        if ilp_label == true_label:
            corr_ilp += 1
            
    accuracy_ilp = corr_ilp / args.run
    print(f"Checked {args.run} ILP outputs. Accuracy: {accuracy_ilp*100:.2f}%")
    
    max_diff = 0.0
    for i in range(10):
        ilp_pt_file = f"{ilp_dir}/orbit_{args.model}{args.act}{args.n}k_Lm{args.Lm}_Sw{args.Sw}_Csw20_bypass_noqbp_comp_part_run{i}_pl.out"
        ilp_ct_file = f"{ilp_dir}/orbit_{args.model}{args.act}{args.n}k_Lm{args.Lm}_Sw{args.Sw}_Csw15_bypass_noqbp_comp_part_run{i}.out"
        diff = get_ptct_diff(ilp_ct_file, ilp_pt_file)
        if diff > max_diff:
            max_diff = diff
        print(f"Run {i} Diff: {diff:.8f}")
    print(f"Max PT-CT Diff over 10 runs: {max_diff:.8f}")
    