from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

from .iterative_partition import solve_partition
from .qbp_manager import QBPManager
from .openevolve_backend import run_compile_openevolve

import time
import sys

def orbit_core(dag: Tdag, le: LatencyEstimator, params: Params):
    ilp_times = dict()
    
    qbp_name = "_".join([
        "qbps_cache",
        f"L{params.lvl_ub}",
        f"LB{params.bts_lb}",
        f"Sf{params.Sf}",
        f"Sw{params.Sw}",
        f"PB{params.placement_backend}",
        "16k" if params.netname.endswith("16k") else "64k"
    ])
    if params.resilience_profile is not None:
        qbp_name += f"_res_{params.resilience_mode}_{params.resilience_profile.fingerprint}"
    
    start_time = time.time()
    if (
        params.placement_backend == "openevolve"
        and params.openevolve_iterations > 0
        and params.openevolve_harness == "compile"
        and getattr(params, "openevolve_compile_hints", None) is None
    ):
        params.openevolve_compile_hints = run_compile_openevolve(dag, le, params)
    ilp_times['OpenEvolve Compile Harness Time'] = time.time() - start_time

    start_time = time.time()
    qbp_manager = QBPManager(params, le)
    if params.reqbp:
        qbp_manager.load_qbps(qbp_name)
    ilp_times['QBP Manager Init Time'] = time.time() - start_time
    
    start_time = time.time()
    partition_result = solve_partition(dag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
    ilp_times['Placement-QBP Time'] = time.time() - start_time
    ilp_times['ILP-QBP Time'] = ilp_times['Placement-QBP Time']
    if partition_result is None:
        print("Error: No solution found for whole DAG", file=sys.stderr)
        return None, ilp_times
    io_to_assign, io_to_cost = partition_result
    
    if io_to_assign is None or io_to_cost is None:
        print("Error: No solution found for whole DAG", file=sys.stderr)
        return None, ilp_times
    
    if params.reqbp:
        start_time = time.time()
        qbp_manager.save_qbps(qbp_name)
        ilp_times['QBP Save Time'] = time.time() - start_time
    
    start_time = time.time()
    final_io_choice = None
    final_cost = None
    for (in_lvl, in_scale), out_to_cost in io_to_cost.items():
        for (out_lvl, out_scale), cost in out_to_cost.items():
            if final_cost is None or final_cost > cost:
                final_cost = cost
                final_io_choice = (in_lvl, in_scale, out_lvl, out_scale)
            
    if final_io_choice is None:
        print("Error: No solution found for whole DAG", file=sys.stderr)
        return None, ilp_times
    
    assign = io_to_assign[final_io_choice[:2]][final_io_choice[2:]]
    ilp_times["Final Assignment Time"] = time.time() - start_time

    print(f"Final Choice: input (l,s)=({final_io_choice[0]},{final_io_choice[1]}), output (l,s)=({final_io_choice[2]},{final_io_choice[3]}), cost={final_cost}")
    print(f"Final cost (aggregated partition cost): {final_cost/1000000:.3f} sec.")
    return assign, ilp_times
