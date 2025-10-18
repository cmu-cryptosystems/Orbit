from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params
from ...utils.fix_random import set_global_seed
from .region_partition import region_partition
from .resbm_core import resbm_core
import time

import argparse

def run(input_file: str, output_file: str, params: Params):
    le = LatencyEstimator(params)
    og_dag = build_from_mlir(input_file, params)
    og_dag.squash_ops(['rescale', 'upscale', 'modswitch', 'bootstrap'])
    print(f"Built original DAG with {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
    addition_squash(og_dag)
    print(f"After Squashing, DAG has {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
    comp_dag, og_to_comp = auto_compress(og_dag, is_ignore_weight=True)
    print(f"After Compression, DAG has {len(comp_dag.nodes)} nodes and {len(comp_dag.edges)} edges.")

    # # visualize(comp_dag, f"{output_file}_comp.svg")
    
    regions, node_to_region, edge_to_region = region_partition(comp_dag)
    for node, region in node_to_region.items():
        comp_dag.nodes[node]['comment'] = f"{region}"
    # visualize(comp_dag, f"{output_file}_regions.svg")
    # visualize_with_region(comp_dag, f"{output_file}_regions_cluster.svg")
    # # for reg_id, reg_dag in regions.items():
    # #     visualize(reg_dag, f"{output_file}_region_{reg_id}.svg")

    final_assign = resbm_core(comp_dag, regions, node_to_region, edge_to_region, le)
    
    assign_lat = estimate_assign(final_assign, le)
    print(f"Final assignment latency: {assign_lat/1000000:.3f} sec.")
    # assign_op_lat, assign_op_lat_bd = estimate_op_assign(final_assign, le)
    # print(f"Final assignment operation latency: {assign_op_lat/1000000:.3f} sec.")
    # for op, cost in assign_op_lat_bd.items():
    #     print(f"  {op}: {cost/1000000:.3f} sec.")
    
    fdag = decode_assign(final_assign, og_dag, og_to_comp)
    
    final_lat = estimate_tdag_latency(fdag, le)
    print(f"Final tdag latency: {final_lat/1000000:.3f} sec. (accurate estimation)")
    _, final_lat_bd_costs = estimate_tdag_latency_breakdown(fdag, le)
    print("Final tdag latency breakdown (in sec):")
    for op, stats in final_lat_bd_costs.items():
        total_op_cost = sum(stats.values())
        print(f"  {op}: {total_op_cost/1000000:.3f} sec.")
    tdag_to_mlir(fdag, output_file)

def main():
    set_global_seed(42)
    parser = argparse.ArgumentParser(description='ReSBM Optimizer')
    parser.add_argument('--inputfile', type=str, required=True, help='Input MLIR file')
    parser.add_argument('--outputfile', type=str, required=True, help='Output MLIR file')
    parser.add_argument('--costjson', type=str, required=True, help='Cost model JSON file')
    parser.add_argument('-Lm', '--maxlevel', type=int, default=None, help='Maximum level budget (overrides costjson if specified)')
    parser.add_argument('-Lub', '--btsupperbound', type=int, default=None, help='Maximum Bootstrapping target level (overrides costjson if specified)')
    parser.add_argument('-Lb', '--btslevel', type=int, default=None, help='Bootstrapping level (overrides costjson if specified)')
    parser.add_argument('-Sw', '--waterscale', type=int, default=40, help='Waterline scale')
    parser.add_argument('-Sf', '--rescale', type=int, default=None, help='Rescaling factor (overrides costjson if specified)')
    
    args = parser.parse_args()
    params = Params(args.costjson, "ReSBM", mode="compile", Sw=args.waterscale)
    if args.maxlevel is not None:
        params.lvl_ub = args.maxlevel
    if args.btsupperbound is not None:
        params.bts_ub = args.btsupperbound
    if args.btslevel is not None:
        params.bts_lb = args.btslevel
    if args.rescale is not None:
        params.Sf = args.rescale
    
    start_time = time.time()
    run(args.inputfile, args.outputfile, params)
    end_time = time.time()
    print(f"ReSBM Compilation time: {end_time - start_time:.3f} sec.")

if __name__ == "__main__":
    main()