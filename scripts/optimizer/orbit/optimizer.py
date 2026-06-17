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


def _optimize_and_emit_mlir(og_dag, output_file: str, params: Params, le: LatencyEstimator, timestamps: dict) -> None:
    """Shared Orbit path: compress → ILP → decode → write MLIR (``og_dag`` already loaded and squashed)."""
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
        comp_dag = og_dag.copy_tdag()
        og_to_comp = {node: node for node in og_dag.nodes}

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


def run(input_file: str, output_file: str, params: Params):
    le = LatencyEstimator(params)

    timestamps = dict()

    start_time = time.time()
    og_dag = build_from_mlir(input_file, params)
    og_dag.squash_ops(['rescale', 'upscale', 'modswitch', 'bootstrap'])
    print(f"Built original DAG with {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
    timestamps['DAG Load Time'] = time.time() - start_time

    _optimize_and_emit_mlir(og_dag, output_file, params, le, timestamps)


def run_from_rotom(manifest_path: str, output_file: str, params: Params) -> None:
    """Same as :func:`run` but builds the initial Tdag from a Rotom circuit manifest (``build_from_rotom``)."""
    le = LatencyEstimator(params)

    timestamps = dict()

    start_time = time.time()
    og_dag = build_from_rotom(manifest_path, params)
    og_dag.squash_ops(['rescale', 'upscale', 'modswitch', 'bootstrap'])
    print(f"Built original DAG from Rotom with {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
    timestamps['DAG Load Time'] = time.time() - start_time

    _optimize_and_emit_mlir(og_dag, output_file, params, le, timestamps)

def main():
    set_global_seed(42)
    parser = argparse.ArgumentParser(description='Orbit Optimizer')
    parser.add_argument('--inputfile', type=str, required=True, help='Input MLIR file')
    parser.add_argument('--outputfile', type=str, required=True, help='Output MLIR file')
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
                    netname=netname)
    if args.maxlevel is not None:
        params.lvl_ub = args.maxlevel
    if args.btsupperbound is not None:
        params.bts_ub = args.btsupperbound
    if args.btslevel is not None:
        params.bts_lb = args.btslevel
    if args.rescale is not None:
        params.Sf = args.rescale
    
    # make sure output_file directory exists
    os.makedirs(os.path.dirname(args.outputfile), exist_ok=True)

    run(args.inputfile, args.outputfile, params)

if __name__ == "__main__":
    main()