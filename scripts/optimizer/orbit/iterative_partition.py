from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

from .siso_partition import rdag_siso_partition, handle_bypass
from .qbp_manager import QBPManager
from .ilp_core import solve_ilp

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import sys
import time

@dataclass
class BoundaryState:
    level: int
    scale: int
    error_abs: float
    cost: float
    path: list[tuple[Tdag, Assign]] = field(default_factory=list)


def solve_partition(dag: Tdag, qbp_manager: QBPManager, prev_cost: dict, le: LatencyEstimator, params: Params):
    if params.use_bounded_resilience_decomposition():
        return solve_partition_bounded_dp(dag, qbp_manager, prev_cost, le, params)

    if not params.part:
        qbp_manager.add_qbp(dag, prev_cost)
        this_io_to_cost = qbp_manager.get_qbp_cost(dag.name)
        this_io_to_assign = qbp_manager.get_qbp_assign(dag)
        return this_io_to_assign, this_io_to_cost
    
    # print(f"Enter solve_partition, PDAG #{dag.name}, prev_cost: {prev_cost}")
    
    is_whole_circ = (len(prev_cost) == 1 and -1 in prev_cost)
    pdags = rdag_siso_partition(dag, 100)
    
    if len(pdags) == 1:
        start_time = time.time()
        main_pdag, bypass_pdag = handle_bypass(dag, params.bpsdepth)
        if bypass_pdag is None:
            print(f"PDAG #{dag.name} Basic, DAG size: {len(dag.nodes)} nodes")
            qbp_manager.add_qbp(dag, prev_cost)
        else:
            # bypass handling
            main_pdag_parts = rdag_siso_partition(main_pdag, 100)
            if len(main_pdag_parts) == 1 and len(bypass_pdag.nodes) == 3:
                print(f"PDAG #{dag.name} Basic, DAG size: {len(dag.nodes)} nodes")
                qbp_manager.add_qbp(dag, prev_cost)
            else:
                print(f"PDAG #{dag.name} with Bypass, Main PDAG size: {len(main_pdag.nodes)} nodes, Bypass PDAG size: {len(bypass_pdag.nodes)} nodes")
                solve_partition(main_pdag, qbp_manager, prev_cost, le, params)
                qbp_manager.add_qbp_bypass(dag, main_pdag, bypass_pdag, prev_cost)
        print(f"Partition solving time for PDAG #{dag.name}: {time.time() - start_time:.2f} seconds")
        if is_whole_circ:
            this_io_to_cost = qbp_manager.get_qbp_cost(dag.name)
            this_io_to_assign = qbp_manager.get_qbp_assign(dag)
            return this_io_to_assign, this_io_to_cost
        return
    
    # for i in range(len(pdags)):
    #     visualize(pdags[i], f"visualize/debug_ilp_pdag_{pdags[i].name}.svg")
    
    dag_io_to_cost = dict()
    dag_io_to_assign = dict()
    this_prev_cost = prev_cost.copy()
    in_budget_choices = []
    
    for i in range(len(pdags)):
        pdag = pdags[i]
        solve_partition(pdag, qbp_manager, this_prev_cost, le, params)
        this_io_to_cost = qbp_manager.get_qbp_cost(pdag.name)
        
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
    for out_lvl, out_scl_to_cost in this_prev_cost.items():
        for out_scl, final_cost in out_scl_to_cost.items():
            this_choice_lvl = out_lvl
            this_choice_scl = out_scl
            final_assign = dict()
            
            for i in range(len(pdags)-1, -1, -1):
                pdag = pdags[i]
                assert (this_choice_lvl, this_choice_scl) in in_budget_choices[i], f"Error: No choice found for out-level {this_choice_lvl} and out-scale {this_choice_scl} at pdag #{pdag.name}"
                in_choice_lvl = in_budget_choices[i][(this_choice_lvl, this_choice_scl)][0]
                in_choice_scl = in_budget_choices[i][(this_choice_lvl, this_choice_scl)][1]
                # get assignment
                this_assign = qbp_manager.get_qbp_assign(pdag)[(in_choice_lvl, in_choice_scl)][(this_choice_lvl, this_choice_scl)]
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
    if not is_whole_circ:
        qbp_manager.add_qbp_existing(dag, dag_io_to_cost, dag_io_to_assign)
    else:
        # for whole dag, no need to add to qbp_manager
        return dag_io_to_assign, dag_io_to_cost


def _initial_boundary_states(prev_cost: dict, params: Params) -> list[BoundaryState]:
    states = []
    input_error = params.resilience_error_model["input_error_abs"]
    for level, scale_to_cost in prev_cost.items():
        for scale, cost in scale_to_cost.items():
            states.append(
                BoundaryState(
                    level=level,
                    scale=scale,
                    error_abs=input_error,
                    cost=cost,
                )
            )
    return states


def _error_bucket(error_abs: float, params: Params) -> int:
    buckets = max(1, int(params.resilience_error_buckets))
    max_error = float(params.resilience_error_model["max_error_abs"])
    if params.resilience_profile is not None:
        tau_values = [
            float(constraint.tau_abs)
            for constraint in params.resilience_profile.constraints
            if constraint.tau_abs is not None
        ]
        if tau_values:
            max_error = max(max(tau_values), params.resilience_error_model["bootstrap_error_abs"])
    if max_error <= 0:
        return 0
    normalized = min(max(float(error_abs), 0.0), max_error) / max_error
    return min(buckets - 1, int(math.floor(normalized * buckets)))


def _state_sort_key(state: BoundaryState, params: Params) -> tuple[float, int, int, int]:
    return (
        state.cost,
        _error_bucket(state.error_abs, params),
        state.level,
        state.scale,
    )


def _prune_boundary_states(states: list[BoundaryState], params: Params) -> tuple[list[BoundaryState], int]:
    if not states:
        return [], 0

    best_by_bucket: dict[tuple[int, int, int], BoundaryState] = {}
    for state in states:
        key = (state.level, state.scale, _error_bucket(state.error_abs, params))
        current = best_by_bucket.get(key)
        if current is None or state.cost < current.cost:
            best_by_bucket[key] = state

    candidates = sorted(best_by_bucket.values(), key=lambda state: _state_sort_key(state, params))
    nondominated: list[BoundaryState] = []
    for state in candidates:
        dominated = False
        for kept in nondominated:
            if (
                kept.level == state.level
                and kept.scale == state.scale
                and kept.cost <= state.cost
                and kept.error_abs <= state.error_abs
            ):
                dominated = True
                break
        if not dominated:
            nondominated.append(state)

    limit = max(1, int(params.resilience_max_boundary_states))
    kept = sorted(nondominated, key=lambda state: _state_sort_key(state, params))[:limit]
    return kept, len(states) - len(kept)


def _merge_state_path(dag: Tdag, state: BoundaryState) -> Assign:
    final_assign: dict[str, dict] = {}
    path = state.path
    names = [
        'v_lvl_out',
        'v_scl_out',
        'v_lvl_in',
        'v_scl_in',
        'e_lvl_out',
        'e_scl_out',
        'v_err_in',
        'v_err_out',
        'e_err_out',
    ]
    for i, (pdag, assign) in enumerate(path):
        for name in names:
            asn = getattr(assign, name)
            if name not in final_assign:
                final_assign[name] = dict()
            for key, val in asn.items():
                if key in pdag.inputs and name in ['v_lvl_in', 'v_scl_in', 'v_err_in'] and i != 0:
                    continue
                if key in pdag.outputs and name in ['v_lvl_out', 'v_scl_out', 'v_err_out'] and i != len(path) - 1:
                    continue
                if key not in final_assign[name]:
                    final_assign[name][key] = val
                else:
                    old = final_assign[name][key]
                    if isinstance(old, float) or isinstance(val, float):
                        assert abs(float(old) - float(val)) < 1e-6, (
                            f"Conflict assignment for {key} in {name} PDAG {pdag.name}: {old} vs {val}"
                        )
                    else:
                        assert old == val, (
                            f"Conflict assignment for {key} in {name} PDAG {pdag.name}: {old} vs {val}"
                        )
    return Assign.from_dict(dag, final_assign)


def _solve_window_candidates(
    pdag: Tdag,
    states: list[BoundaryState],
    le: LatencyEstimator,
    params: Params,
) -> tuple[list[BoundaryState], int]:
    if not states:
        return [], 0

    tasks = []
    for state_index, state in enumerate(states):
        for out_lvl in range(1, params.lvl_ub + 1):
            io_budget = {
                'in_lvl': state.level,
                'in_scl': state.scale,
                'out_lvl': out_lvl,
                'in_error_abs': state.error_abs,
            }
            tasks.append((state_index, io_budget))

    if not tasks:
        return [], 0

    worker_count = min(max(1, params.threads), len(tasks))
    ilp_threads = max(1, int(math.ceil(params.threads / worker_count)))
    candidates: list[BoundaryState] = []
    failed = 0

    def run_task(task):
        state_index, io_budget = task
        task_name = (
            f"Task (BoundedDP-{pdag.name} "
            f"in_lvl={io_budget['in_lvl']}_in_scl={io_budget['in_scl']}_"
            f"out_lvl={io_budget['out_lvl']})"
        )
        return state_index, solve_ilp(pdag, io_budget, le, task_name, ilp_threads, params)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            try:
                state_index, (assign, assign_cost) = future.result()
            except Exception as exc:
                failed += 1
                print(f"Warning: bounded-DP window solve failed for PDAG #{pdag.name}: {exc}", file=sys.stderr)
                continue
            if assign is None or assign_cost is None:
                failed += 1
                continue

            state = states[state_index]
            pdag_out = list(pdag.outputs)[0]
            out_error = assign.v_err_out.get(pdag_out, state.error_abs)
            candidates.append(
                BoundaryState(
                    level=assign.v_lvl_out[pdag_out],
                    scale=assign.v_scl_out[pdag_out],
                    error_abs=float(out_error),
                    cost=state.cost + assign_cost,
                    path=state.path + [(pdag, assign)],
                )
            )

    return candidates, failed


def solve_partition_bounded_dp(
    dag: Tdag,
    qbp_manager: QBPManager,
    prev_cost: dict,
    le: LatencyEstimator,
    params: Params,
):
    start_time = time.time()
    threshold = max(2, int(params.resilience_decompose_threshold))
    pdags = rdag_siso_partition(dag, threshold)
    is_whole_circ = len(prev_cost) == 1 and -1 in prev_cost
    report = {
        "mode": "bounded-dp",
        "threshold": threshold,
        "windows": len(pdags),
        "initial_states": 0,
        "ilp_tasks_attempted": 0,
        "ilp_tasks_failed": 0,
        "states_generated": 0,
        "states_pruned": 0,
        "final_states": 0,
        "elapsed_sec": 0.0,
    }

    states = _initial_boundary_states(prev_cost, params)
    report["initial_states"] = len(states)
    print(
        f"Resilience decomposition: mode=bounded-dp, windows={len(pdags)}, "
        f"threshold={threshold}, max_boundary_states={params.resilience_max_boundary_states}, "
        f"error_buckets={params.resilience_error_buckets}"
    )

    for index, pdag in enumerate(pdags):
        print(
            f"Bounded-DP window {index + 1}/{len(pdags)} for PDAG #{pdag.name}: "
            f"{len(pdag.nodes)} nodes, input_states={len(states)}"
        )
        report["ilp_tasks_attempted"] += len(states) * params.lvl_ub
        candidates, failed = _solve_window_candidates(pdag, states, le, params)
        report["ilp_tasks_failed"] += failed
        report["states_generated"] += len(candidates)
        states, pruned = _prune_boundary_states(candidates, params)
        report["states_pruned"] += pruned
        print(
            f"Bounded-DP window {index + 1}/{len(pdags)} result: "
            f"generated={len(candidates)}, kept={len(states)}, pruned={pruned}, failed_tasks={failed}"
        )
        if not states:
            print(
                f"Error: No valid bounded-DP states remain after PDAG #{pdag.name}",
                file=sys.stderr,
            )
            report["elapsed_sec"] = time.time() - start_time
            params.resilience_decomposition_report = report
            return None, None

    report["final_states"] = len(states)
    report["elapsed_sec"] = time.time() - start_time
    params.resilience_decomposition_report = report
    print(
        "Resilience decomposition report: "
        f"mode=bounded-dp, windows={report['windows']}, "
        f"initial_states={report['initial_states']}, "
        f"states_generated={report['states_generated']}, "
        f"states_pruned={report['states_pruned']}, "
        f"final_states={report['final_states']}, "
        f"ilp_tasks_attempted={report['ilp_tasks_attempted']}, "
        f"ilp_tasks_failed={report['ilp_tasks_failed']}, "
        f"elapsed_sec={report['elapsed_sec']:.3f}"
    )

    dag_io_to_cost: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    dag_io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]] = {}
    for state in states:
        if not state.path:
            continue
        first_pdag, first_assign = state.path[0]
        last_pdag, last_assign = state.path[-1]
        pdag_in = list(first_pdag.inputs)[0]
        pdag_out = list(last_pdag.outputs)[0]
        in_key = (
            first_assign.v_lvl_in.get(pdag_in, state.level),
            first_assign.v_scl_in.get(pdag_in, params.Sw),
        )
        out_key = (
            last_assign.v_lvl_out[pdag_out],
            last_assign.v_scl_out[pdag_out],
        )
        if -1 in prev_cost:
            in_key = (-1, params.Sw)
        final_assign = _merge_state_path(dag, state)
        start_cost = 0
        if in_key[0] in prev_cost and in_key[1] in prev_cost[in_key[0]]:
            start_cost = prev_cost[in_key[0]][in_key[1]]
        real_final_cost = state.cost - start_cost
        current = dag_io_to_cost.get(in_key, {}).get(out_key)
        if current is None or real_final_cost < current:
            dag_io_to_cost.setdefault(in_key, {})[out_key] = real_final_cost
            dag_io_to_assign.setdefault(in_key, {})[out_key] = final_assign

    if not is_whole_circ:
        qbp_manager.add_qbp_existing(dag, dag_io_to_cost, dag_io_to_assign)
    else:
        return dag_io_to_assign, dag_io_to_cost
