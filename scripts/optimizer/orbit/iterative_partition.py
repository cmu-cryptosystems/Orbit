from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

from .siso_partition import rdag_siso_partition, handle_bypass
from .qbp_manager import QBPManager

import sys
import time

def solve_partition(dag: Tdag, qbp_manager: QBPManager, prev_cost: dict, le: LatencyEstimator, params: Params):
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