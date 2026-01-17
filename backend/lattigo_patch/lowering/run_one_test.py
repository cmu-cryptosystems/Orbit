import argparse
import os
import subprocess
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orbit ILP Solver')

    parser.add_argument('--model', type=str, required=True, choices=["ResNet", "AlexNet", "SqueezeNet", "MobileNet", "VGG16"], help='Model architecture')
    parser.add_argument('--act', type=str, required=True, choices=["ReLU", "SiLU"], help='Activation function')
    parser.add_argument('--n', type=int, required=True, choices=[16, 64], help='CKKS Vector Size')
    parser.add_argument('--Lm', type=int, required=True, help='Maximum Level Budget')
    parser.add_argument('--Sw', type=int, required=True, help='Waterline Scale')
    parser.add_argument('--Csw', type=int, default=None, help='Constant Waterline Scale')
    parser.add_argument('--cmt', type=str, default="bypass_noqbp_comp_part", help='Comments at the end of orbit results')
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
    # dacapo_mlir = f"{args.model}{tmp_act}{args.n}.{args.Sw}.LATTIGO.CPU{args.n}.earth"
    # dacapo_mlir = f"{args.model}.{args.Sw}.earth"
    dacapo_mlir = benchmark
    
    # MLIR 
    orbit_mlir_dir = f"../../BtsPlc/mlirs_output/orbit/{args.model}/{args.Sw}/{args.act}/{this_n}/{orbit_mlir}.mlir"
    # dacapo_mlir_dir = f"mlirs_old/results/dacapo/{args.model}/{args.Sw}/{tmp_act}/{this_n}/{dacapo_mlir}.mlir"
    # dacapo_mlir_dir = f"../../fresh-dacapo/examples/traced/{dacapo_mlir}.mlir"
    dacapo_mlir_dir = f"../../dacaporion/dacapo/examples/traced/{dacapo_mlir}.mlir"
    
    
    # CONS, INPUT
    inp_cst_dir = f"inputs/{args.n}k/{args.model.lower()}/{args.act.lower()}"
    
    inp_file = f"{inp_cst_dir}/inputs/input{args.run}.txt"
    # cst_file = f"{inp_cst_dir}/_hecate_{args.model}{args.act}.cst"
    # cst_file = f"{inp_cst_dir}/_hecate_{args.model}.cst"
    cst_file = f"mlir_const/_hecate_{benchmark}.cst"
    # cst_file = f"../../fresh-dacapo/examples/traced/_hecate_{args.model}.cst"
    # cst_file = f"../../dacaporion/dacapo/examples/traced/_hecate_{benchmark}.cst"
    
    result_dir = f"execution_res/{args.n}/{args.model}/{args.act}/"
    os.makedirs(result_dir, exist_ok=True)
    
    # OUTPUT
    orbit_out_file = f"../{result_dir}{orbit_mlir}_run{args.run}{args.lcmt}.out"
    dacapo_out_file = f"../{result_dir}{dacapo_mlir}_run{args.run}{args.lcmt}.out"

    # LOG
    orbit_log_file = f"{result_dir}{orbit_mlir}_run{args.run}{args.lcmt}.log"
    dacapo_log_file = f"{result_dir}{dacapo_mlir}_run{args.run}{args.lcmt}.log"

    # # Prof file
    # orbit_prof_file = f"{result_dir}{orbit_mlir}_run{args.run}{args.lcmt}.prof"

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
    
    # cmds_dacapo = [
    #     "go", "run", "./fhe",
    #     "-n", "16384" if args.n == 16 else "65536",
    #     "-maxLevel", str(args.Lm),
    #     "-bootstrapMinLevel", "3",
    #     "-bootstrapMaxLevel", str(args.Lm),
    #     "-mlir", dacapo_mlir_dir,
    #     "-cons", cst_file,
    #     "-input", inp_file,
    #     "-output", dacapo_out_file
    # ]   
    # if args.plain:
    #     cmds_dacapo.append("-heMode=false")
    
    # with open(dacapo_log_file, "w", buffering=1) as stdout_file:
    #     # Run the command
    #     process = subprocess.Popen(
    #         cmds_dacapo,
    #         stdout=stdout_file
    #     )

    #     process.wait()