import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReSBM')
    parser.add_argument('--model', type=str, required=True, choices=["ResNet", "AlexNet", "SqueezeNet", "MobileNet", "VGG16"], help='Model architecture')
    parser.add_argument('--act', type=str, required=True, choices=["ReLU", "SiLU"], help='Activation function')
    parser.add_argument('--n', type=int, required=True, choices=[16, 64], help='CKKS Vector Size')
    parser.add_argument('--Lm', type=int, required=True, help='Maximum Level Budget')
    parser.add_argument('--Sw', type=int, required=True, help='Waterline Scale')
    parser.add_argument('--sim-vari', action='store_true', help='Use simulated variadic bootstrapping cost model')
    
    args = parser.parse_args()
    benchmark = args.model+args.act+str(args.n)+"k"
    this_n = args.Lm if args.Lm != 16 else args.n
    if args.sim_vari:
        assert args.n == 64
        assert args.Lm == 16
        costs = f"cost_models/profiled_LATTIGONEW_CPU{args.n}k_3_{args.Lm}_sim_vari.json"
    else:
        costs = f"cost_models/profiled_LATTIGONEW_CPU{args.n}k_3_{args.Lm}.json"
    
    result_dir = f"mlirs_output/ReSBM/{args.model}/{args.Sw}/{args.act}/{this_n}/"
    outname = f"ReSBM_{benchmark}_Lm{args.Lm}_Sw{args.Sw}" + ("_simvari" if args.sim_vari else "")
    
    os.makedirs(result_dir, exist_ok=True)
    
    cmds = [
        "python", "-u", "-m",
        "scripts.optimizer.ReSBM.optimizer",
        "--inputfile", f"mlirs_input/{benchmark}.mlir",
        "--outputfile", f"{result_dir}{outname}.mlir",
        "--costjson", costs,
        "--maxlevel", str(args.Lm),
        "--btsupperbound", str(args.Lm),
        "--waterscale", str(args.Sw),
    ]
        
    with open(f"{result_dir}{outname}.txt", "w", buffering=1) as stdout_file, \
        open(f"{result_dir}{outname}.err", "w", buffering=1) as stderr_file:
        process = subprocess.Popen(
            cmds,
            stdout=stdout_file,
            stderr=stderr_file
        )
        process.wait()