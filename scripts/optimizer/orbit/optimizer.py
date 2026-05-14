from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params
from ...utils.fix_random import set_global_seed

from .orbit_core import orbit_core

import argparse
import time
import os

def run(input_file: str, output_file: str, params: Params, input_format: str = "mlir"):
    le = LatencyEstimator(params)

    timestamps = dict()

    start_time = time.time()
    if input_format == "rotom":
        og_dag = build_from_rotom(input_file, params)
    else:
        og_dag = build_from_mlir(input_file, params)
    og_dag.squash_ops(['rescale', 'upscale', 'modswitch', 'bootstrap'])
    print(f"Built original DAG with {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
    if params.resilience_profile is not None:
        print(f"Loaded resilience profile: {params.resilience_profile.describe()}")
        print(f"Resilience mode: {params.resilience_mode}")
        print(f"Resilience constraint policy: {params.resilience_constraint_policy}")
        print(f"Resilience decomposition: {params.resilience_decomposition}")
    timestamps['DAG Load Time'] = time.time() - start_time
    
    if params.comp:
        start_time = time.time()
        addition_squash(og_dag)
        if params.bpsdepth is not None:
            addition_desquash_bypass(og_dag)
        print(f"After Squashing, DAG has {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
        comp_dag, og_to_comp = auto_compress(og_dag, is_ignore_weight=True)
        print(f"After Compression, DAG has {len(comp_dag.nodes)} nodes and {len(comp_dag.edges)} edges.")
        timestamps['DAG Compression Time'] = time.time() - start_time
    else:
        split_constants(og_dag)
        comp_dag = og_dag.copy_tdag()
        og_to_comp = {node: node for node in og_dag.nodes}

    if params.resilience_profile is not None:
        report = params.resilience_profile.match_report(comp_dag)
        params.resilience_match_report = report
        print(params.resilience_profile.format_match_report(report))
        if report["unmatched_targets"]:
            preview = ", ".join(str(target) for target in report["unmatched_targets"][:5])
            suffix = "..." if len(report["unmatched_targets"]) > 5 else ""
            print(f"Unmatched resilience targets: {preview}{suffix}")
        if report["matched_nodes"] == 0 and not params.allow_empty_resilience_match:
            raise RuntimeError(
                "Resilience profile matched zero compressed TDAG nodes. "
                "Use --allow-empty-resilience-match only to debug old profiles."
            )
    
    assign, ilp_times = orbit_core(comp_dag, le, params)
    for k, v in ilp_times.items():
        timestamps[k] = v
    
    assign_lat = estimate_assign(assign, le)
    print(f"Final assignment latency: {assign_lat/1000000:.3f} sec.")
    
    start_time = time.time()
    fdag = decode_assign(assign, og_dag, og_to_comp)
    timestamps['DAG Decode Time'] = time.time() - start_time
    
    final_lat = estimate_tdag_latency(fdag, le)
    print(f"Final tdag latency: {final_lat/1000000:.3f} sec. (accurate estimation)")
    final_lat_bd_num, final_lat_bd_costs = estimate_tdag_latency_breakdown(fdag, le)
    print("Final tdag latency breakdown (number of operations):")
    for op, stats in sorted(final_lat_bd_num.items()):
        total_op_num = sum(stats.values())
        print(f"  {op}: {total_op_num} ops.")
    print("=" * 40)
    print("Final tdag latency breakdown (in sec):")
    for op, stats in sorted(final_lat_bd_costs.items()):
        total_op_cost = sum(stats.values())
        print(f"  {op}: {total_op_cost/1000000:.3f} sec.")
    print("=" * 40)
    
    start_time = time.time()
    tdag_to_mlir(fdag, output_file)
    timestamps['DAG Write Time'] = time.time() - start_time
    
    print(f"Orbit Compilation time: {sum(timestamps.values()):.3f} sec.")
    print("Timestamps breakdown:")
    for k, v in timestamps.items():
        print(f"  '{k}': {v:.3f} sec.")
    print("=" * 40)

def main():
    set_global_seed(42)
    parser = argparse.ArgumentParser(description='Orbit Optimizer')
    parser.add_argument('--inputfile', type=str, required=True,
                        help='Input file: .mlir file or Rotom manifest .json')
    parser.add_argument('--outputfile', type=str, required=True, help='Output MLIR file')
    parser.add_argument('--input-format', type=str, default='auto',
                        choices=['auto', 'mlir', 'rotom'],
                        help='Input format (default: auto-detect from file extension)')
    parser.add_argument('--costjson', type=str, required=True, help='Cost model JSON file')
    parser.add_argument('--maxlevel', type=int, default=None, help='Maximum level budget (overrides costjson if specified)')
    parser.add_argument('--btsupperbound', type=int, default=None, help='Maximum Bootstrapping target level (overrides costjson if specified)')
    parser.add_argument('--btslevel', type=int, default=None, help='Bootstrapping level (overrides costjson if specified)')
    parser.add_argument('-Sw', '--waterscale', type=int, default=40, help='Waterline scale')
    parser.add_argument('-Csw', '--constantscale', type=int, default=None, help='Constant scale (overrides waterline scale if specified)')
    parser.add_argument('-Sf', '--rescale', type=int, default=None, help='Rescaling factor (overrides costjson if specified)')
    parser.add_argument('--nobypass', action='store_true', help='Disable Bypass handling')
    parser.add_argument('--no-compress', action='store_true', help='Disable DAG compression')
    parser.add_argument('--no-partition', action='store_true', help='Disable SISO partitioning')
    parser.add_argument('--enable-reqbp', action='store_true', help='Enable QBP cross-bench reusing')
    parser.add_argument('--bypass-dep', type=int, default=15, help='Bypass dependency level (default: 15)')
    parser.add_argument('--threads', type=int, default=16, help='Number of threads (default: 16)')
    parser.add_argument('--netname', type=str, default="", help='Network name for qbp reusing purposes (default: mlirs_input/<netname>.mlir)')
    parser.add_argument(
        '--resilience-profile',
        type=str,
        default=None,
        help='ckks-robustness-profiler JSON or Orbit resilience constraints JSON',
    )
    parser.add_argument(
        '--resilience-mode',
        choices=['waterline', 'error-state'],
        default='waterline',
        help='How resilience constraints affect placement (default: waterline)',
    )
    parser.add_argument(
        '--allow-empty-resilience-match',
        action='store_true',
        help='Allow a loaded resilience profile to match zero TDAG nodes',
    )
    parser.add_argument(
        '--resilience-decomposition',
        choices=['off', 'bounded-dp'],
        default='off',
        help='Optional decomposition strategy for resilience error-state search',
    )
    parser.add_argument(
        '--resilience-decompose-threshold',
        type=int,
        default=32,
        help='SISO window size threshold for bounded-DP resilience decomposition',
    )
    parser.add_argument(
        '--resilience-max-boundary-states',
        type=int,
        default=8,
        help='Maximum boundary states retained after each bounded-DP window',
    )
    parser.add_argument(
        '--resilience-error-buckets',
        type=int,
        default=8,
        help='Number of error buckets used for bounded-DP boundary pruning',
    )
    parser.add_argument(
        '--ilp-task-time-limit-sec',
        type=float,
        default=0.0,
        help='Per-ILP Gurobi time limit in seconds; 0 means no limit',
    )
    parser.add_argument(
        '--resilience-constraint-policy',
        choices=['relax-only', 'hard-tau'],
        default='relax-only',
        help=(
            'How profile tolerances constrain placement. relax-only uses the '
            'profile only to lower local scale requirements; hard-tau also '
            'enforces tau_abs as an error upper bound.'
        ),
    )
    
    args = parser.parse_args()
    if args.nobypass:
        bypass_dep = None
    else:
        bypass_dep = args.bypass_dep
        
    if args.netname != "":
        netname = args.netname
    else:
        netname = os.path.splitext(os.path.basename(args.inputfile))[0]
    params = Params(args.costjson, "Orbit", mode="compile", 
                    Sw=args.waterscale, CSw=args.constantscale, bpsdepth=bypass_dep, threads=args.threads, 
                    comp=not args.no_compress, part=not args.no_partition, reqbp=args.enable_reqbp, 
                    netname=netname, resilience_profile=args.resilience_profile,
                    resilience_mode=args.resilience_mode,
                    allow_empty_resilience_match=args.allow_empty_resilience_match,
                    resilience_decomposition=args.resilience_decomposition,
                    resilience_decompose_threshold=args.resilience_decompose_threshold,
                    resilience_max_boundary_states=args.resilience_max_boundary_states,
                    resilience_error_buckets=args.resilience_error_buckets,
                    ilp_task_time_limit_sec=args.ilp_task_time_limit_sec,
                    resilience_constraint_policy=args.resilience_constraint_policy)
    if args.maxlevel is not None:
        params.lvl_ub = args.maxlevel
    if args.btsupperbound is not None:
        params.bts_ub = args.btsupperbound
    if args.btslevel is not None:
        params.bts_lb = args.btslevel
    if args.rescale is not None:
        params.Sf = args.rescale
    
    # Auto-detect input format from file extension
    input_format = args.input_format
    if input_format == 'auto':
        if args.inputfile.endswith('.json'):
            input_format = 'rotom'
        else:
            input_format = 'mlir'

    # make sure output_file directory exists
    os.makedirs(os.path.dirname(args.outputfile), exist_ok=True)

    run(args.inputfile, args.outputfile, params, input_format=input_format)

if __name__ == "__main__":
    main()
