import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Saturn ILP Solver')
    parser.add_argument('--model', type=str, required=True, choices=["ResNet", "AlexNet", "SqueezeNet", "MobileNet", "VGG16", "CompPart"], help='Model architecture')
    parser.add_argument('--act', type=str, required=True, choices=["ReLU", "SiLU"], help='Activation function')
    parser.add_argument('--n', type=int, required=True, choices=[16, 64], help='CKKS Vector Size')
    parser.add_argument('--Lm', type=int, required=True, help='Maximum Level Budget')
    parser.add_argument('--Sw', type=int, required=True, help='Waterline Scale')
    parser.add_argument('--Csw', type=int, required=False, default=None, help='Constant Scale')
    parser.add_argument('--nobypass', action='store_true', help='Disable Bypass handling')
    parser.add_argument('--qbp', action='store_true', help='Enable QBP cross-bench reusing')
    parser.add_argument('--nocomp', action='store_true', help='Disable Compression')
    parser.add_argument('--nopart', action='store_true', help='Disable Partitioning')
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
    
    result_dir = f"mlirs_output/saturn/{args.model}/{args.Sw}/{args.act}/{this_n}/"
    outname = f"saturn_{benchmark}_Lm{args.Lm}_Sw{args.Sw}" + (f"_Csw{args.Csw}" if args.Csw is not None else "") + ("_nobypass" if args.nobypass else "_bypass") + ("_qbp" if args.qbp else "_noqbp") + ("_nocomp" if args.nocomp else "_comp") + ("_nopart" if args.nopart else "_part") + ("_simvari" if args.sim_vari else "")
    
    os.makedirs(result_dir, exist_ok=True)
    
    cmds = [
        "python", "-u", "-m",
        "scripts.optimizer.saturn.optimizer",
        "--inputfile", f"mlirs_input/{benchmark}.mlir",
        "--outputfile", f"{result_dir}{outname}.mlir",
        "--costjson", costs,
        "--maxlevel", str(args.Lm),
        "--waterscale", str(args.Sw),
    ]
    if args.Csw is not None:
        cmds += ["--constantscale", str(args.Csw)]
    if args.nobypass:
        cmds.append("--nobypass")
    if args.nocomp:
        cmds.append("--no-compress")
    if args.nopart:
        cmds.append("--no-partition")
    if args.qbp:
        cmds.append("--enable-reqbp")
        
    with open(f"{result_dir}{outname}.txt", "w", buffering=1) as stdout_file, \
        open(f"{result_dir}{outname}.err", "w", buffering=1) as stderr_file:
        process = subprocess.Popen(
            cmds,
            stdout=stdout_file,
            stderr=stderr_file
        )
        process.wait()