from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

from .siso_partition import rdag_siso_partition, handle_bypass
from .qbp_manager import QBPManager

import sys
import time
import os

def solve_partition(dag: Tdag, qbp_manager: QBPManager, prev_cost: dict, le: LatencyEstimator, params: Params):
    progress_start = time.time()
    _openevolve_partition_progress(
        params,
        f"enter dag={dag.name} nodes={len(dag.nodes)} edges={len(dag.edges)} "
        f"prev_states={_prev_cost_state_count(prev_cost)} part={params.part}",
    )
    if not params.part:
        qbp_manager.add_qbp(dag, prev_cost)
        this_io_to_cost = qbp_manager.get_qbp_cost(dag.name)
        this_io_to_assign = qbp_manager.get_qbp_assign(dag)
        _openevolve_partition_progress(
            params,
            f"exit no-part dag={dag.name} input_states={len(this_io_to_cost)} "
            f"elapsed={time.time() - progress_start:.2f}s",
        )
        return this_io_to_assign, this_io_to_cost
    
    # print(f"Enter solve_partition, PDAG #{dag.name}, prev_cost: {prev_cost}")
    
    is_whole_circ = (len(prev_cost) == 1 and -1 in prev_cost)
    pdags = rdag_siso_partition(dag, 100)
    _openevolve_partition_progress(
        params,
        f"partitioned dag={dag.name} pdags={len(pdags)} "
        f"elapsed={time.time() - progress_start:.2f}s",
    )
    
    if len(pdags) == 1:
        start_time = time.time()
        main_pdag, bypass_pdag = handle_bypass(dag, params.bpsdepth)
        if bypass_pdag is None:
            print(f"PDAG #{dag.name} Basic, DAG size: {len(dag.nodes)} nodes", flush=True)
            _openevolve_partition_progress(params, f"add_qbp basic dag={dag.name}")
            qbp_manager.add_qbp(dag, prev_cost)
        else:
            # bypass handling
            main_pdag_parts = rdag_siso_partition(main_pdag, 100)
            if len(main_pdag_parts) == 1 and len(bypass_pdag.nodes) == 3:
                print(f"PDAG #{dag.name} Basic, DAG size: {len(dag.nodes)} nodes", flush=True)
                _openevolve_partition_progress(params, f"add_qbp basic-bypass-small dag={dag.name}")
                qbp_manager.add_qbp(dag, prev_cost)
            else:
                print(f"PDAG #{dag.name} with Bypass, Main PDAG size: {len(main_pdag.nodes)} nodes, Bypass PDAG size: {len(bypass_pdag.nodes)} nodes", flush=True)
                _openevolve_partition_progress(
                    params,
                    f"bypass recurse dag={dag.name} main={main_pdag.name} "
                    f"main_nodes={len(main_pdag.nodes)} bypass_nodes={len(bypass_pdag.nodes)}",
                )
                main_partition_result = solve_partition(main_pdag, qbp_manager, prev_cost, le, params)
                _register_partition_result(qbp_manager, main_pdag, main_partition_result)
                _openevolve_partition_progress(params, f"add_qbp_bypass dag={dag.name}")
                qbp_manager.add_qbp_bypass(dag, main_pdag, bypass_pdag, prev_cost)
        print(f"Partition solving time for PDAG #{dag.name}: {time.time() - start_time:.2f} seconds", flush=True)
        _openevolve_partition_progress(
            params,
            f"exit single dag={dag.name} elapsed={time.time() - progress_start:.2f}s",
        )
        # orbit_core unpacks this pair; a bare return here used to yield None and crash.
        this_io_to_cost = qbp_manager.get_qbp_cost(dag.name)
        this_io_to_assign = qbp_manager.get_qbp_assign(dag)
        return this_io_to_assign, this_io_to_cost
    
    # for i in range(len(pdags)):
    #     visualize(pdags[i], f"visualize/debug_ilp_pdag_{pdags[i].name}.svg")
    
    dag_io_to_cost = dict()
    dag_io_to_assign = dict()
    this_prev_cost = prev_cost.copy()
    in_budget_choices = []
    
    for i in range(len(pdags)):
        pdag = pdags[i]
        step_start = time.time()
        _openevolve_partition_progress(
            params,
            f"subpartition start dag={dag.name} index={i+1}/{len(pdags)} "
            f"pdag={pdag.name} nodes={len(pdag.nodes)} prev_states={_prev_cost_state_count(this_prev_cost)}",
        )
        solve_partition(pdag, qbp_manager, this_prev_cost, le, params)
        this_io_to_cost = qbp_manager.get_qbp_cost(pdag.name)
        _openevolve_partition_progress(
            params,
            f"subpartition qbp done dag={dag.name} index={i+1}/{len(pdags)} "
            f"pdag={pdag.name} input_states={len(this_io_to_cost)} "
            f"elapsed={time.time() - step_start:.2f}s",
        )
        
        # re-stating the initial prev_cost
        if len(this_prev_cost) == 1 and -1 in this_prev_cost:
            this_prev_cost = dict()
            for l in range(1, params.lvl_ub + 1):
                this_prev_cost[l] = {params.Sw: 0}
        
        cur_cost = dict()
        in_budget_choice = dict()
        # Core DP
        for (in_lvl, in_scl), out_to_cost in this_io_to_cost.items():
            # exclude qbps not for current partition input budget
            if in_lvl not in this_prev_cost or in_scl not in this_prev_cost[in_lvl]:
                continue
            for (out_lvl, out_scl), cost in out_to_cost.items():
                this_cost = this_prev_cost[in_lvl][in_scl] + cost
                if out_lvl not in cur_cost or out_scl not in cur_cost[out_lvl] or cur_cost[out_lvl][out_scl] > this_cost:
                    if out_lvl not in cur_cost:
                        cur_cost[out_lvl] = dict()
                    cur_cost[out_lvl][out_scl] = this_cost
                    in_budget_choice[(out_lvl, out_scl)] = (in_lvl, in_scl)

        in_budget_choices.append(in_budget_choice)
        this_prev_cost = cur_cost
        _openevolve_partition_progress(
            params,
            f"subpartition dp done dag={dag.name} index={i+1}/{len(pdags)} "
            f"cur_states={_prev_cost_state_count(this_prev_cost)} "
            f"elapsed={time.time() - step_start:.2f}s",
        )
        # iterative method pruning
        if i == len(pdags) - 1:
            # final partition: no pruning
            break
        new_prev_cost = dict()
        
        abs_tol = 0.8 * pdag.get_full_size() * le.lin_op_lmaps['rescale_single'][1]
        
        for out_lvl, scl_to_cost in this_prev_cost.items():
            # find the minimal output scale
            # with cost <= minimal cost + abs_tol 
            min_cost = min(scl_to_cost.values()) + abs_tol
            min_scl = None
            for out_scl, cost in scl_to_cost.items():
                if cost <= min_cost:
                    if min_scl is None or out_scl < min_scl:
                        min_scl = out_scl
            new_prev_cost[out_lvl] = {min_scl: scl_to_cost[min_scl]}
        this_prev_cost = new_prev_cost
        _openevolve_partition_progress(
            params,
            f"subpartition prune done dag={dag.name} index={i+1}/{len(pdags)} "
            f"states={_prev_cost_state_count(this_prev_cost)} "
            f"elapsed={time.time() - step_start:.2f}s",
        )
        
        if len(this_prev_cost) == 0:
            break
    
    if len(this_prev_cost) == 0:
        print(f"Error: No valid partitioning found for DAG #{dag.name} with input budgets {prev_cost}", file=sys.stderr)
        return
    
    if len(prev_cost) == 1 and -1 in prev_cost:
        prev_cost = dict()
        for l in range(1, params.lvl_ub + 1):
            prev_cost[l] = {params.Sw: 0}
    
    # merge results from all partitions
    merge_start = time.time()
    _openevolve_partition_progress(
        params,
        f"merge start dag={dag.name} pdags={len(pdags)} "
        f"output_states={_prev_cost_state_count(this_prev_cost)}",
    )
    path_candidates = []
    for out_lvl, out_scl_to_cost in this_prev_cost.items():
        for out_scl, final_cost in out_scl_to_cost.items():
            this_choice_lvl = out_lvl
            this_choice_scl = out_scl
            final_assign = dict()
            path_records_reversed = []
            
            for i in range(len(pdags)-1, -1, -1):
                pdag = pdags[i]
                assert (this_choice_lvl, this_choice_scl) in in_budget_choices[i], f"Error: No choice found for out-level {this_choice_lvl} and out-scale {this_choice_scl} at pdag #{pdag.name}"
                out_choice_lvl = this_choice_lvl
                out_choice_scl = this_choice_scl
                in_choice_lvl = in_budget_choices[i][(this_choice_lvl, this_choice_scl)][0]
                in_choice_scl = in_budget_choices[i][(this_choice_lvl, this_choice_scl)][1]
                # get assignment
                this_assign = qbp_manager.get_qbp_assign(pdag)[(in_choice_lvl, in_choice_scl)][(this_choice_lvl, this_choice_scl)]
                part_cost = (
                    qbp_manager.get_qbp_cost(pdag.name)
                    .get((in_choice_lvl, in_choice_scl), {})
                    .get((out_choice_lvl, out_choice_scl), None)
                )
                path_records_reversed.append({
                    "pdag_name": pdag.name,
                    "partition_index": i,
                    "in_key": [int(in_choice_lvl), int(in_choice_scl)],
                    "out_key": [int(out_choice_lvl), int(out_choice_scl)],
                    "group_key": {
                        "in_lvl": int(in_choice_lvl),
                        "in_scl": int(in_choice_scl),
                        "maino_v": "",
                        "main_dag_size": 0,
                    },
                    "cost_usec": float(part_cost) if part_cost is not None else None,
                    "bootstrap_count": float(_count_assign_bootstraps(this_assign, params)),
                })
                this_choice_lvl = in_choice_lvl
                this_choice_scl = in_choice_scl
                
                # merge assignment
                for name in ['v_lvl_out', 'v_scl_out', 'v_lvl_in', 'v_scl_in', 'e_lvl_out', 'e_scl_out']:
                    asn = getattr(this_assign, name)
                    if name not in final_assign:
                        final_assign[name] = dict()
                    for v, val in asn.items():
                        if v in pdags[i].inputs and name in ['v_lvl_in', 'v_scl_in']:
                            if i != 0:
                                continue
                        elif v in pdags[i].outputs and name in ['v_lvl_out', 'v_scl_out']:
                            if i != len(pdags) - 1:
                                continue
                        if v not in final_assign[name]:
                            final_assign[name][v] = val
                        else:
                            assert final_assign[name][v] == val, f"i={i}, len_pdags={len(pdags)}, Conflict assignment for node/edge {v} in {name} PDAG {pdag.name}, inputs {pdag.inputs}, outputs {pdag.outputs}: {final_assign[name][v]} vs {val}"
            
            # update final assign
            final_assign_update = Assign.from_dict(dag, final_assign)
            
            assert this_choice_lvl in prev_cost, f"Error: Final choice level {this_choice_lvl} does not belong to the input budgets {prev_cost} for PDAG {dag.name}"
            l_scl_to_cost = prev_cost[this_choice_lvl]
            assert this_choice_scl in l_scl_to_cost, f"Error: Final choice scale {this_choice_scl} does not match input scale {l_scl_to_cost} for PDAG {dag.name}"
            
            real_final_cost = final_cost - l_scl_to_cost[this_choice_scl]
            
            if (this_choice_lvl, this_choice_scl) not in dag_io_to_cost:
                dag_io_to_cost[(this_choice_lvl, this_choice_scl)] = {(out_lvl, out_scl): real_final_cost}
                dag_io_to_assign[(this_choice_lvl, this_choice_scl)] = {(out_lvl, out_scl): final_assign_update}
            else:
                dag_io_to_cost[(this_choice_lvl, this_choice_scl)][(out_lvl, out_scl)] = real_final_cost
                dag_io_to_assign[(this_choice_lvl, this_choice_scl)][(out_lvl, out_scl)] = final_assign_update
            path_candidates.append({
                "dag_name": dag.name,
                "whole_circuit": bool(is_whole_circ),
                "input_key": [int(this_choice_lvl), int(this_choice_scl)],
                "output_key": [int(out_lvl), int(out_scl)],
                "final_cost_usec": float(real_final_cost),
                "partition_count": len(pdags),
                "partitions": list(reversed(path_records_reversed)),
            })
    if (
        path_candidates
        and (
            getattr(params, "openevolve_evaluating_candidate", False)
            or getattr(params, "openevolve_collect_diagnostics", False)
        )
    ):
        path_candidates.sort(key=lambda item: float(item.get("final_cost_usec", float("inf"))))
        qbp_manager.openevolve_selected_path_records.extend(path_candidates[:8])
    if not is_whole_circ:
        qbp_manager.add_qbp_existing(dag, dag_io_to_cost, dag_io_to_assign)
        _openevolve_partition_progress(
            params,
            f"merge exit dag={dag.name} elapsed={time.time() - merge_start:.2f}s "
            f"total={time.time() - progress_start:.2f}s",
        )
    else:
        # for whole dag, no need to add to qbp_manager
        _openevolve_partition_progress(
            params,
            f"merge exit whole dag={dag.name} elapsed={time.time() - merge_start:.2f}s "
            f"total={time.time() - progress_start:.2f}s",
        )
        return dag_io_to_assign, dag_io_to_cost


def _register_partition_result(qbp_manager: QBPManager, pdag: Tdag, partition_result):
    """Register whole-subgraph partition results needed by enclosing bypass merges."""
    if pdag.name in qbp_manager.pdag_name_to_qbp:
        return
    if partition_result is None:
        return
    io_to_assign, io_to_cost = partition_result
    if io_to_assign is None or io_to_cost is None:
        return
    qbp_manager.add_qbp_existing(pdag, io_to_cost, io_to_assign)


def _count_assign_bootstraps(assign: Assign, params: Params) -> int:
    total = 0
    for node in assign.tdag.nodes:
        if assign.tdag.nodes[node].get("op") == "constant":
            continue
        if (
            node in assign.v_lvl_in
            and node in assign.v_scl_in
            and node in assign.v_lvl_out
            and node in assign.v_scl_out
            and not params.check_res(
                assign.v_lvl_in[node],
                assign.v_scl_in[node],
                assign.v_lvl_out[node],
                assign.v_scl_out[node],
            )
        ):
            total += 1
    for pred, dst in assign.tdag.edges:
        if assign.tdag.nodes[pred].get("op") == "constant":
            continue
        if (
            (pred, dst) in assign.e_lvl_out
            and (pred, dst) in assign.e_scl_out
            and pred in assign.v_lvl_out
            and pred in assign.v_scl_out
            and not params.check_res(
                assign.v_lvl_out[pred],
                assign.v_scl_out[pred],
                assign.e_lvl_out[(pred, dst)],
                assign.e_scl_out[(pred, dst)],
            )
        ):
            total += 1
    return total


def _openevolve_partition_progress(params: Params, message: str):
    if getattr(params, "placement_backend", "") != "openevolve":
        return
    raw = os.environ.get("ORBIT_OPENEVOLVE_PARTITION_PROGRESS", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return
    if not raw and not getattr(params, "openevolve_collect_diagnostics", False):
        return
    print(f"[OpenEvolve partition] {message}", file=sys.__stderr__, flush=True)


def _prev_cost_state_count(prev_cost: dict) -> int:
    total = 0
    for value in prev_cost.values():
        if isinstance(value, dict):
            total += len(value)
        else:
            total += 1
    return total
