from __future__ import annotations

from typing import Any

from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params


def add_ilp_linear_cost(tdag: Tdag, vp: Any, le: LatencyEstimator):
    cost_rescale_s = le.lin_op_lmaps['rescale_single']
    cost_bts_s = le.lin_op_lmaps['bootstrap_single']
    vp.rescale_cost = cost_rescale_s[1]

    # add node rescale/bootstrap costs
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
        vp.total_cost.append(tdag.nodes[v]['weight'] * cost_bts_s[1] * vp.var_use(v, 'b'))
        vp.total_cost.append(tdag.nodes[v]['weight'] * cost_rescale_s[1] * vp.var_use(v, 'r'))
    # add edge rescale costs
    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        vp.total_cost.append(tdag.edges[u, v]['weight'] * cost_rescale_s[1] * vp.var_use((u, v), 'r'))
    for v in tdag.nodes:
        single_cnt, double_cnt = tdag.get_v_weights(v)
        if single_cnt > 0:
            this_cost = le.lin_op_lmaps[tdag.nodes[v]['op'] + '_single']
            vp.total_cost.append(single_cnt * (this_cost[0] * vp.var_lvl(v, 'in') + this_cost[1]))
        if double_cnt > 0:
            this_cost = le.lin_op_lmaps[tdag.nodes[v]['op'] + '_double']
            vp.total_cost.append(double_cnt * (this_cost[0] * vp.var_lvl(v, 'in') + this_cost[1]))


def _var_sol(x: Any) -> float:
    return float(x.X)


def decode_ilp_sol(tdag: Tdag, vp: Any) -> Assign:
    assign = Assign(tdag)
    params = vp.params
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'input':
            # input nodes, need to store in-level/scale
            assign.v_lvl_in[v] = round(_var_sol(vp.var_lvl(v, 'in')))
            assign.v_scl_in[v] = round(_var_sol(vp.var_scl(v, 'in')))
            assert assign.v_scl_in[v] >= params.Sw, f"Node {v} input scale {assign.v_scl_in[v]} below Sw={params.Sw}"
        assign.v_lvl_out[v] = round(_var_sol(vp.var_lvl(v, 'out')))
        assign.v_scl_out[v] = round(_var_sol(vp.var_scl(v, 'out')))
        if tdag.nodes[v]['op'] != 'constant':
            assert assign.v_scl_out[v] >= params.Sw, f"Node {v} output scale {assign.v_scl_out[v]} below Sw={params.Sw}"

    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        assign.e_lvl_out[(u, v)] = round(_var_sol(vp.var_lvl((u, v), 'out')))
        assign.e_scl_out[(u, v)] = round(_var_sol(vp.var_scl((u, v), 'out')))
        assert assign.e_scl_out[(u, v)] >= params.Sw, f"Edge ({u},{v}) output scale {assign.e_scl_out[(u,v)]} below Sw={params.Sw}"

    return assign


def solve_ilp(
    tdag: Tdag,
    io_budgets: dict,
    le: LatencyEstimator,
    task_name: str,
    num_threads: int,
    params: Params,
) -> tuple[Assign | None, float | None]:
    try:
        import gurobipy as gp
        from .ilp_gurobi import (
            VarPool,
            add_ilp_constraints,
            add_ilp_io_budgets,
            gurobi_has_solution,
            solve_ilp_core,
            solve_ilp_core_bypass,
        )
    except ImportError as e:
        raise ImportError(
            "gurobipy is required to run Orbit. Install it with: pip install gurobipy"
        ) from e

    model = gp.Model(task_name)
    vp = VarPool(tdag, params, model)

    add_ilp_constraints(tdag, vp)
    add_ilp_linear_cost(tdag, vp, le)
    add_ilp_io_budgets(tdag, vp, io_budgets)
    if 'maino_v' in io_budgets:
        solve_ilp_core_bypass(tdag, vp, io_budgets, num_threads)
    else:
        solve_ilp_core(vp, num_threads)

    if not gurobi_has_solution(model):
        return None, None
    assign = decode_ilp_sol(tdag, vp)
    if 'maino_v' in io_budgets:
        v_main_o = io_budgets['maino_v']
        mo_lvl = assign.v_lvl_in[v_main_o]
        mo_scl = assign.v_scl_in[v_main_o]
        assert (mo_lvl, mo_scl) in io_budgets['main_qbp_cost'], f"Main output (lvl, scl)=({mo_lvl}, {mo_scl}) not in provided QBP costs"

    assign_cost = estimate_assign(assign, le)
    return assign, assign_cost
