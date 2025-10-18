from .latency_estimator import LatencyEstimator
from ..assignment.assignment import Assign
from ..tdag.tdag import Tdag
from ..params.params import Params

def estimate_assign(assign: Assign, le: LatencyEstimator) -> float:
    tdag = assign.tdag
    total_cost = 0
    
    for v in tdag.nodes:
        op = tdag.nodes[v]['op']
        if op == 'constant':
            continue
        
        v_lin, v_sin = assign.get_v_in_lvl_scl(v)
        v_lout = assign.v_lvl_out.get(v, None)
        v_sout = assign.v_scl_out.get(v, None)
        
        # operation cost
        if op not in ['input', 'output', 'dummy']:
            single_cnt, double_cnt = tdag.get_v_weights(v)
            if single_cnt > 0:
                total_cost += le.op_lmaps[f"{op}_single"][v_lin] * single_cnt
            if double_cnt > 0:
                total_cost += le.op_lmaps[f"{op}_double"][v_lin] * double_cnt
        
        # vertex cost
        total_cost += tdag.nodes[v]['weight'] * le.resbts_cost(v_lin, v_sin, v_lout, v_sout)
        
        # edge cost
        for u in tdag.predecessors(v):
            if tdag.nodes[u]['op'] == 'constant':
                continue
            e_lout = assign.e_lvl_out.get((u, v), None)
            e_sout = assign.e_scl_out.get((u, v), None)
            u_lin = assign.v_lvl_out.get(u, None)
            u_sin = assign.v_scl_out.get(u, None)
            total_cost += tdag.edges[u, v]['weight'] * le.resbts_cost(u_lin, u_sin, e_lout, e_sout)

    return total_cost

def estimate_op_assign(assign: Assign, le: LatencyEstimator) -> float:
    tdag = assign.tdag
    cost_dict = dict()
    total_cost = 0
    
    
    for v in tdag.nodes:
        op = tdag.nodes[v]['op']
        if op == 'constant':
            continue
        
        v_lin, _ = assign.get_v_in_lvl_scl(v)
        
        # operation cost
        if op not in ['input', 'output', 'dummy']:
            single_cnt, double_cnt = tdag.get_v_weights(v)
            if single_cnt > 0:
                total_cost += le.op_lmaps[f"{op}_single"][v_lin] * single_cnt
                if f"{op}_single" not in cost_dict:
                    cost_dict[f"{op}_single"] = 0
                cost_dict[f"{op}_single"] += le.op_lmaps[f"{op}_single"][v_lin] * single_cnt
            if double_cnt > 0:
                total_cost += le.op_lmaps[f"{op}_double"][v_lin] * double_cnt
                if f"{op}_double" not in cost_dict:
                    cost_dict[f"{op}_double"] = 0
                cost_dict[f"{op}_double"] += le.op_lmaps[f"{op}_double"][v_lin] * double_cnt

    return total_cost, cost_dict