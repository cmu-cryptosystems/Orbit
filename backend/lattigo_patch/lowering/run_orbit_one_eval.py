import argparse
import os
import subprocess
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orbit Evaluation Runner')
    # This script is tailored for running orbit-generated MLIRs for different benchmarks and configurations
    # For running Dacapo, Orion, ReSBM, changes may be needed

    parser.add_argument('--model', type=str, required=True, choices=["ResNet", "AlexNet", "SqueezeNet", "MobileNet", "VGG16", "CompPart"], help='Model architecture')
    parser.add_argument('--act', type=str, required=True, choices=["ReLU", "SiLU"], help='Activation function')
    parser.add_argument('--n', type=int, required=True, choices=[16, 64], help='CKKS Vector Size')
    parser.add_argument('--Lm', type=int, required=True, help='Maximum Level Budget')
    parser.add_argument('--Sw', type=int, required=True, help='Waterline Scale')
    parser.add_argument('--Csw', type=int, default=None, help='Constant Waterline Scale')
    parser.add_argument('--cmt', type=str, default="bypass_noqbp_comp_part", help='Comments at the end of orbit results')
    # cmt for micro-benchmarks: 
    # - without bypass handling: "nobypass_noqbp_comp_part"
    # - with/without compression / partitioning: "bypass_noqbp_<comp|nocomp>_<part|nopart>"
    # - qbp reusing: "bypass_qbp_comp_part"
    parser.add_argument('--lcmt', type=str, default="", help='Comments appended to the log file')
    parser.add_argument('--run', type=int, required=True, choices=range(1000), help='Which input file to run')
    parser.add_argument('--plain', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    if args.plain:
        args.lcmt += "_pl"
    
    benchmark = args.model+args.act+str(args.n)+"k"
    this_n = args.Lm if args.Lm != 16 else args.n
    cpu_name = f"CPU{this_n}" if this_n != 64 else "CPU"
    
    tmp_act = "Silu" if args.act == "SiLU" else "Relu"
    
    if args.Csw is not None:
        orbit_mlir = f"orbit_{benchmark}_Lm{args.Lm}_Sw{args.Sw}_Csw{args.Csw}_{args.cmt}"
    else:
        orbit_mlir = f"orbit_{benchmark}_Lm{args.Lm}_Sw{args.Sw}_{args.cmt}"
        
    # MLIR 
    orbit_mlir_dir = f"../../../mlirs_output/orbit/{args.model}/{args.Sw}/{args.act}/{this_n}/{orbit_mlir}.mlir"
    
    # CONS, INPUT
    if args.model == "CompPart":
        assert args.n == 64, "CompPart only supports n=64k"
        assert args.act == "SiLU", "CompPart only supports SiLU activation"
        # use ResNet input and constants for CompPart
        inp_cst_dir = f"../../../input_data/{args.n}k/resnet/{args.act.lower()}"
        inp_file = f"{inp_cst_dir}/inputs/input{args.run}.txt"
        cst_file = f"../../../input_constants/ResNet{args.act}{args.n}k_hecate.cst"
    else:
        inp_cst_dir = f"../../../input_data/{args.n}k/{args.model.lower()}/{args.act.lower()}"
        inp_file = f"{inp_cst_dir}/inputs/input{args.run}.txt"
        cst_file = f"../../../input_constants/{benchmark}_hecate.cst"
    
    result_dir = f"../../../mlirs_execute/orbit/{args.n}/{args.model}/{args.act}/"
    os.makedirs(result_dir, exist_ok=True)
    
    # OUTPUT
    orbit_out_file = f"../{result_dir}{orbit_mlir}_run{args.run}{args.lcmt}.out"
    # LOG
    orbit_log_file = f"{result_dir}{orbit_mlir}_run{args.run}{args.lcmt}.log"

    cmds_orbit = [
        "go", "run", "./fhe",
        "-n", "16384" if args.n == 16 else "65536",
        "-maxLevel", str(args.Lm),
        "-bootstrapMinLevel", "3",
        "-bootstrapMaxLevel", str(args.Lm),
        "-mlir", orbit_mlir_dir,
        "-cons", cst_file,
        "-input", inp_file,
        "-output", orbit_out_file
    ]
    if args.plain:
        cmds_orbit.append("-heMode=false")
    
    with open(orbit_log_file, "w", buffering=1) as stdout_file:
        # Run the command
        process = subprocess.Popen(
            cmds_orbit,
            stdout=stdout_file
        )

        process.wait()
        # shutil.copyfile("outputs/profile.prof", orbit_prof_file)