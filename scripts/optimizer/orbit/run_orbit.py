import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orbit placement optimizer')
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
    parser.add_argument('--placement-backend', choices=['openevolve', 'ilp'], default='openevolve')
    parser.add_argument('--ilp-solver', choices=['pulp', 'gurobi'], default=None)
    parser.add_argument('--openevolve-config', type=str, default=None)
    parser.add_argument('--openevolve-output-dir', type=str, default=None)
    parser.add_argument('--openevolve-iterations', type=int, default=0)
    parser.add_argument('--openevolve-seed', type=int, default=42)
    parser.add_argument('--openevolve-keep-workdir', action='store_true')
    parser.add_argument('--openevolve-provider', choices=['gemini', 'openai', 'custom'], default='gemini')
    parser.add_argument('--openevolve-model', type=str, default='gemini-3.1-pro-preview')
    parser.add_argument('--openevolve-api-base', type=str, default=None)
    parser.add_argument('--openevolve-api-key-env', type=str, default='OPENAI_API_KEY')
    parser.add_argument('--openevolve-harness', choices=['compile', 'partition'], default='compile')
    parser.add_argument(
        '--openevolve-eval-suite',
        choices=['toy', 'polybert-sampled', 'polybert-full'],
        default='polybert-sampled',
    )
    parser.add_argument('--openevolve-reference-json', type=str, default=None)
    parser.add_argument('--openevolve-finalists', type=int, default=3)
    parser.add_argument('--resilience-profile', type=str, default=None)
    parser.add_argument('--resilience-mode', choices=['waterline', 'error-state'], default='waterline')
    parser.add_argument('--allow-empty-resilience-match', action='store_true')
    parser.add_argument('--resilience-decomposition', choices=['off', 'bounded-dp'], default='off')
    parser.add_argument('--resilience-decompose-threshold', type=int, default=32)
    parser.add_argument('--resilience-max-boundary-states', type=int, default=8)
    parser.add_argument('--resilience-error-buckets', type=int, default=8)
    parser.add_argument('--resilience-constraint-policy', choices=['relax-only', 'hard-tau'], default='relax-only')
    parser.add_argument('--ilp-task-time-limit-sec', type=float, default=0.0)
    
    args = parser.parse_args()
    benchmark = args.model+args.act+str(args.n)+"k"
    this_n = args.Lm if args.Lm != 16 else args.n
    if args.sim_vari:
        assert args.n == 64
        assert args.Lm == 16
        costs = f"cost_models/profiled_LATTIGONEW_CPU{args.n}k_3_{args.Lm}_sim_vari.json"
    else:
        costs = f"cost_models/profiled_LATTIGONEW_CPU{args.n}k_3_{args.Lm}.json"
    
    result_dir = f"mlirs_output/orbit/{args.model}/{args.Sw}/{args.act}/{this_n}/"
    outname = (
        f"orbit_{benchmark}_Lm{args.Lm}_Sw{args.Sw}"
        + (f"_Csw{args.Csw}" if args.Csw is not None else "")
        + ("_nobypass" if args.nobypass else "_bypass")
        + ("_qbp" if args.qbp else "_noqbp")
        + ("_nocomp" if args.nocomp else "_comp")
        + ("_nopart" if args.nopart else "_part")
        + ("_simvari" if args.sim_vari else "")
        + f"_{args.placement_backend}"
        + (f"_{args.ilp_solver}" if args.placement_backend == "ilp" and args.ilp_solver else "")
        + (f"_oe{args.openevolve_iterations}" if args.placement_backend == "openevolve" else "")
        + ("_resilience" if args.resilience_profile else "")
    )
    
    os.makedirs(result_dir, exist_ok=True)
    
    cmds = [
        "python3", "-u", "-m",
        "scripts.optimizer.orbit.optimizer",
        "--inputfile", f"mlirs_input/{benchmark}.mlir",
        "--outputfile", f"{result_dir}{outname}.mlir",
        "--costjson", costs,
        "--maxlevel", str(args.Lm),
        "--waterscale", str(args.Sw),
        "--placement-backend", args.placement_backend,
    ]
    if args.placement_backend == "ilp" and args.ilp_solver:
        cmds += ["--ilp-solver", args.ilp_solver]
    if args.placement_backend == "openevolve":
        cmds += ["--openevolve-iterations", str(args.openevolve_iterations)]
        cmds += ["--openevolve-seed", str(args.openevolve_seed)]
        cmds += ["--openevolve-provider", args.openevolve_provider]
        cmds += ["--openevolve-model", args.openevolve_model]
        cmds += ["--openevolve-api-key-env", args.openevolve_api_key_env]
        cmds += ["--openevolve-harness", args.openevolve_harness]
        cmds += ["--openevolve-eval-suite", args.openevolve_eval_suite]
        cmds += ["--openevolve-finalists", str(args.openevolve_finalists)]
        if args.openevolve_reference_json:
            cmds += ["--openevolve-reference-json", args.openevolve_reference_json]
        if args.openevolve_api_base:
            cmds += ["--openevolve-api-base", args.openevolve_api_base]
        if args.openevolve_config:
            cmds += ["--openevolve-config", args.openevolve_config]
        if args.openevolve_output_dir:
            cmds += ["--openevolve-output-dir", args.openevolve_output_dir]
        if args.openevolve_keep_workdir:
            cmds.append("--openevolve-keep-workdir")
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
    if args.resilience_profile:
        cmds += ["--resilience-profile", args.resilience_profile]
        cmds += ["--resilience-mode", args.resilience_mode]
        cmds += ["--resilience-decomposition", args.resilience_decomposition]
        cmds += ["--resilience-decompose-threshold", str(args.resilience_decompose_threshold)]
        cmds += ["--resilience-max-boundary-states", str(args.resilience_max_boundary_states)]
        cmds += ["--resilience-error-buckets", str(args.resilience_error_buckets)]
        cmds += ["--resilience-constraint-policy", args.resilience_constraint_policy]
        cmds += ["--ilp-task-time-limit-sec", str(args.ilp_task_time_limit_sec)]
    if args.allow_empty_resilience_match:
        cmds.append("--allow-empty-resilience-match")
        
    with open(f"{result_dir}{outname}.txt", "w", buffering=1) as stdout_file, \
        open(f"{result_dir}{outname}.err", "w", buffering=1) as stderr_file:
        process = subprocess.Popen(
            cmds,
            stdout=stdout_file,
            stderr=stderr_file
        )
        exit_code = process.wait()
        if exit_code != 0:
            with open(f"{result_dir}{outname}.err", "r") as err_file:
                error_msg = err_file.read()
            print(f"Subprocess exited with code {exit_code}.\n    Error message:\n{error_msg}")
