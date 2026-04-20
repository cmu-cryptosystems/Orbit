from __future__ import annotations

import re
from typing import Any, Union

from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None  # type: ignore


def _pulp_safe_name(s: str, max_len: int = 240) -> str:
    """PuLP warns if problem/constraint names contain spaces or some punctuation."""
    t = re.sub(r"\s+", "_", str(s).strip())
    t = re.sub(r"[^\w.\-]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return (t or "orbit_ilp")[:max_len]


def _pulp_r_upper(params: Params, smax: int) -> int:
    """Conservative upper bound on rescale-use integer variables (required by CBC)."""
    return max(128, params.lvl_ub * (smax // max(params.Sf, 1) + 4))


class PulpVarPool:
    """Variable pool for PuLP (CBC) — mirrors Gurobi :class:`VarPool` with explicit bounds on integers."""

    def __init__(self, tdag: Tdag, params: Params, model: Any):
        if pulp is None:
            raise ImportError("PuLP is required for ilp_solver='pulp'. Install with: pip install pulp")
        self.params = params
        self.Smax = params.Sf + 2 * params.Sw
        self.model = model
        self.r_ub = _pulp_r_upper(params, self.Smax)
        bts_lb = params.bts_lb
        sf = params.Sf
        # Big-M values for indicator linearization (Gurobi addGenConstrIndicator replacement)
        self.M1 = float(self.Smax + sf * max(0, bts_lb - 1) + sf * params.lvl_ub + 1000.0)
        self.M2 = float(params.bts_lb + params.lvl_ub + 100.0)
        self.M3 = float(max(self.Smax - sf, sf - params.Sw) + 100.0)
        self.M4 = float(params.lvl_ub + self.r_ub + 1000.0)
        self.M5 = float(self.Smax + sf * self.r_ub + 10000.0)

        self.vars: dict[str, Any] = {}
        self.total_cost: list[Any] = []
        self.rescale_cost = None

        for v in tdag.nodes:
            if tdag.nodes[v]['op'] == 'constant':
                continue
            self.vars[f"v_lvl_in_{v}"] = pulp.LpVariable(f"v_lvl_in_{v}", lowBound=1, upBound=params.lvl_ub, cat=pulp.LpInteger)
            self.vars[f"v_scl_in_{v}"] = pulp.LpVariable(f"v_scl_in_{v}", lowBound=params.Sw, upBound=self.Smax, cat=pulp.LpInteger)
            self.vars[f"v_lvl_out_{v}"] = pulp.LpVariable(f"v_lvl_out_{v}", lowBound=1, upBound=params.lvl_ub, cat=pulp.LpInteger)
            self.vars[f"v_scl_out_{v}"] = pulp.LpVariable(f"v_scl_out_{v}", lowBound=params.Sw, upBound=self.Smax, cat=pulp.LpInteger)
            self.vars[f"v_use_r_{v}"] = pulp.LpVariable(f"v_use_r_{v}", lowBound=0, upBound=self.r_ub, cat=pulp.LpInteger)
            self.vars[f"v_use_b_{v}"] = pulp.LpVariable(f"v_use_b_{v}", cat=pulp.LpBinary)

        for u, v in tdag.edges:
            if tdag.nodes[u]['op'] == 'constant':
                continue
            edge_label = self.get_edge_label(u, v)
            self.vars[f"e_lvl_in_{edge_label}"] = self.vars[f"v_lvl_out_{u}"]
            self.vars[f"e_scl_in_{edge_label}"] = self.vars[f"v_scl_out_{u}"]
            self.vars[f"e_lvl_out_{edge_label}"] = self.vars[f"v_lvl_in_{v}"]
            if tdag.nodes[v]['op'] == 'mul':
                self.vars[f"e_scl_out_{edge_label}"] = pulp.LpVariable(
                    f"e_scl_out_{edge_label}", lowBound=params.Sw, upBound=self.Smax, cat=pulp.LpInteger
                )
            else:
                self.vars[f"e_scl_out_{edge_label}"] = self.vars[f"v_scl_in_{v}"]
            self.vars[f"e_use_r_{edge_label}"] = pulp.LpVariable(
                f"e_use_r_{edge_label}", lowBound=0, upBound=self.r_ub, cat=pulp.LpInteger
            )

        for u in tdag.nodes:
            if tdag.nodes[u]['op'] != 'constant':
                continue
            assert tdag.out_degree(u) == 1, f"Constant node {u} should have exactly one output"
            v = list(tdag.successors(u))[0]
            self.vars[f"v_lvl_out_{u}"] = self.vars[f"v_lvl_in_{v}"]
            if tdag.nodes[v]['op'] == 'mul':
                self.vars[f"v_scl_out_{u}"] = pulp.LpVariable(
                    f"v_scl_out_{u}", lowBound=params.Csw, upBound=params.Csw, cat=pulp.LpInteger
                )
            else:
                self.vars[f"v_scl_out_{u}"] = self.vars[f"v_scl_in_{v}"]
            edge_label = self.get_edge_label(u, v)
            self.vars[f"e_lvl_out_{edge_label}"] = self.vars[f"v_lvl_out_{u}"]
            self.vars[f"e_scl_out_{edge_label}"] = self.vars[f"v_scl_out_{u}"]

    def get_edge_label(self, u: str, v: str) -> str:
        return f"({u}_{v})"

    def var_lvl(self, v: str | tuple[str, str], type: str) -> Any:
        if isinstance(v, str):
            return self.vars[f"v_lvl_{type}_{v}"]
        edge_label = self.get_edge_label(v[0], v[1])
        return self.vars[f"e_lvl_{type}_{edge_label}"]

    def var_scl(self, v: str | tuple[str, str], type: str) -> Any:
        if isinstance(v, str):
            return self.vars[f"v_scl_{type}_{v}"]
        edge_label = self.get_edge_label(v[0], v[1])
        return self.vars[f"e_scl_{type}_{edge_label}"]

    def var_use(self, v: str | tuple[str, str], type: str) -> Any:
        if isinstance(v, str):
            return self.vars[f"v_use_{type}_{v}"]
        edge_label = self.get_edge_label(v[0], v[1])
        return self.vars[f"e_use_{type}_{edge_label}"]


def add_ilp_constraints_pulp(tdag: Tdag, vp: PulpVarPool):
    """Same semantics as Gurobi :func:`add_ilp_constraints` using Big-M linearization for indicators."""
    prob = vp.model
    params = vp.params
    sf = params.Sf
    bts_lb = params.bts_lb

    for v in tdag.nodes:
        if tdag.nodes[v]['op'] != 'mul':
            continue
        ilist = list(tdag.predecessors(v))
        if len(ilist) == 1:
            i0, i1 = ilist[0], ilist[0]
        else:
            i0, i1 = ilist[0], ilist[1]
        prob += vp.var_scl(v, 'in') == vp.var_scl((i0, v), 'out') + vp.var_scl((i1, v), 'out'), _pulp_safe_name(f"scl_mul_{v}")

    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
        b = vp.var_use(v, 'b')
        s_in, l_in = vp.var_scl(v, 'in'), vp.var_lvl(v, 'in')
        l_out, s_out = vp.var_lvl(v, 'out'), vp.var_scl(v, 'out')
        r = vp.var_use(v, 'r')
        prob += s_in - sf * (l_in - bts_lb + 1) <= vp.M1 * (1 - b), _pulp_safe_name(f"bts_scl_in_{v}")
        prob += (bts_lb + 1) - l_out <= vp.M2 * (1 - b), _pulp_safe_name(f"bts_lvl_{v}")
        prob += sf - s_out <= vp.M3 * (1 - b), _pulp_safe_name(f"bts_scl_out_{v}")
        prob += l_out - l_in + r <= vp.M4 * b, _pulp_safe_name(f"nobts_lvl_{v}")
        prob += s_in - sf * r - s_out <= vp.M5 * b, _pulp_safe_name(f"nobts_scl_{v}")

    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        edge_label = vp.get_edge_label(u, v)
        prob += vp.var_lvl((u, v), 'out') <= vp.var_lvl((u, v), 'in') - vp.var_use((u, v), 'r'), _pulp_safe_name(f"edge_lvl_{edge_label}")
        prob += vp.var_scl((u, v), 'out') >= vp.var_scl((u, v), 'in') - sf * vp.var_use((u, v), 'r'), _pulp_safe_name(f"edge_scl_{edge_label}")


def add_ilp_linear_cost(tdag: Tdag, vp: Union[Any, PulpVarPool], le: LatencyEstimator):
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


def add_ilp_io_budgets_pulp(tdag: Tdag, vp: PulpVarPool, io_budgets):
    prob = vp.model
    params = vp.params
    v_in = list(tdag.inputs)[0]
    v_out = list(tdag.outputs)[0]
    if 'in_lvl' in io_budgets and io_budgets['in_lvl'] >= 0:
        in_lvl = io_budgets['in_lvl']
        prob += vp.var_lvl(v_in, 'in') == in_lvl, _pulp_safe_name(f"input_level_{in_lvl}")
    if 'in_scl' in io_budgets and io_budgets['in_scl'] >= 0:
        in_scl = io_budgets['in_scl']
        prob += vp.var_scl(v_in, 'in') == in_scl, _pulp_safe_name(f"input_scale_{in_scl}")
    if 'out_lvl' in io_budgets and io_budgets['out_lvl'] >= 0:
        out_lvl = io_budgets['out_lvl']
        prob += vp.var_lvl(v_out, 'out') == out_lvl, _pulp_safe_name(f"output_level_{out_lvl}")

    prob += vp.var_scl(v_out, 'out') <= params.Sf * (vp.var_lvl(v_out, 'out') + 1) - 7, _pulp_safe_name("lattigo_output_scale_level_relation")

    vp.total_cost.append(0.2 * vp.rescale_cost * tdag.get_full_size() * vp.var_scl(v_out, 'out'))


def _pulp_cbc_solver(num_threads: int):
    return pulp.PULP_CBC_CMD(msg=0, threads=num_threads, gapRel=0.01)


def solve_ilp_core_pulp(vp: PulpVarPool, num_threads: int):
    prob = vp.model
    prob += pulp.lpSum(vp.total_cost)
    prob.solve(_pulp_cbc_solver(num_threads))


def solve_ilp_core_bypass_pulp(tdag: Tdag, vp: PulpVarPool, io_budgets, num_threads: int):
    prob = vp.model
    v_main_o = io_budgets['maino_v']
    assert v_main_o in tdag.nodes, f"Main output node {v_main_o} not in Tdag {tdag.name}"
    main_dag_size = io_budgets['main_dag_size']
    main_qbp_cost = io_budgets['main_qbp_cost']
    all_out = list(tdag.outputs)[0]
    solver = _pulp_cbc_solver(num_threads)

    min_cost = None
    min_cost_ls = None

    for (mo_lvl, mo_scl), main_cost in main_qbp_cost.items():
        c_lvl = _pulp_safe_name(f"bypass_main_lvl_{mo_lvl}_{mo_scl}")
        c_scl = _pulp_safe_name(f"bypass_main_scl_{mo_lvl}_{mo_scl}")
        prob += vp.var_lvl(v_main_o, 'in') == mo_lvl, c_lvl
        prob += vp.var_scl(v_main_o, 'in') == mo_scl, c_scl
        prob.solve(solver)
        if pulp.LpStatus[prob.status] == 'Optimal':
            s_out = pulp.value(vp.var_scl(all_out, 'out'))
            if s_out is None:
                s_out = 0.0
            ft_cost = 0.2 * vp.rescale_cost * main_dag_size * round(float(s_out))
            obj = float(pulp.value(prob.objective) or 0.0)
            this_cost = obj + main_cost + ft_cost
            if (min_cost is None) or (this_cost < min_cost):
                min_cost = this_cost
                min_cost_ls = (mo_lvl, mo_scl)
        del prob.constraints[c_lvl]
        del prob.constraints[c_scl]

    if min_cost_ls is None:
        return
    prob += vp.var_lvl(v_main_o, 'in') == min_cost_ls[0], _pulp_safe_name("bypass_final_lvl")
    prob += vp.var_scl(v_main_o, 'in') == min_cost_ls[1], _pulp_safe_name("bypass_final_scl")
    prob.solve(solver)


def _var_sol(x: Any, use_pulp: bool) -> float:
    if use_pulp:
        if pulp is None:
            raise RuntimeError("PuLP not available")
        v = pulp.value(x)
        if v is None:
            raise ValueError("Missing PuLP variable value")
        return float(v)
    return float(x.X)


def decode_ilp_sol(tdag: Tdag, vp: Any, *, use_pulp: bool = False) -> Assign:
    assign = Assign(tdag)
    params = vp.params
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'input':
            # input nodes, need to store in-level/scale
            assign.v_lvl_in[v] = round(_var_sol(vp.var_lvl(v, 'in'), use_pulp))
            assign.v_scl_in[v] = round(_var_sol(vp.var_scl(v, 'in'), use_pulp))
            assert assign.v_scl_in[v] >= params.Sw, f"Node {v} input scale {assign.v_scl_in[v]} below Sw={params.Sw}"
        assign.v_lvl_out[v] = round(_var_sol(vp.var_lvl(v, 'out'), use_pulp))
        assign.v_scl_out[v] = round(_var_sol(vp.var_scl(v, 'out'), use_pulp))
        if tdag.nodes[v]['op'] != 'constant':
            assert assign.v_scl_out[v] >= params.Sw, f"Node {v} output scale {assign.v_scl_out[v]} below Sw={params.Sw}"

    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        assign.e_lvl_out[(u, v)] = round(_var_sol(vp.var_lvl((u, v), 'out'), use_pulp))
        assign.e_scl_out[(u, v)] = round(_var_sol(vp.var_scl((u, v), 'out'), use_pulp))
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
    solver = getattr(params, "ilp_solver", "gurobi")
    if solver == "pulp":
        return _solve_ilp_pulp(tdag, io_budgets, le, task_name, num_threads, params)
    return _solve_ilp_gurobi(tdag, io_budgets, le, task_name, num_threads, params)


def _solve_ilp_gurobi(
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
            "gurobipy is required for ilp_solver='gurobi'. Install gurobipy or use ilp_solver='pulp' (PuLP + CBC)."
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
    assign = decode_ilp_sol(tdag, vp, use_pulp=False)
    if 'maino_v' in io_budgets:
        v_main_o = io_budgets['maino_v']
        mo_lvl = assign.v_lvl_in[v_main_o]
        mo_scl = assign.v_scl_in[v_main_o]
        assert (mo_lvl, mo_scl) in io_budgets['main_qbp_cost'], f"Main output (lvl, scl)=({mo_lvl}, {mo_scl}) not in provided QBP costs"

    assign_cost = estimate_assign(assign, le)
    return assign, assign_cost


def _solve_ilp_pulp(
    tdag: Tdag,
    io_budgets: dict,
    le: LatencyEstimator,
    task_name: str,
    num_threads: int,
    params: Params,
) -> tuple[Assign | None, float | None]:
    if pulp is None:
        raise ImportError("PuLP is required for ilp_solver='pulp'. Install with: pip install pulp")
    prob = pulp.LpProblem(_pulp_safe_name(task_name), pulp.LpMinimize)
    vp = PulpVarPool(tdag, params, prob)

    add_ilp_constraints_pulp(tdag, vp)
    add_ilp_linear_cost(tdag, vp, le)
    add_ilp_io_budgets_pulp(tdag, vp, io_budgets)
    if 'maino_v' in io_budgets:
        solve_ilp_core_bypass_pulp(tdag, vp, io_budgets, num_threads)
    else:
        solve_ilp_core_pulp(vp, num_threads)

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, None
    assign = decode_ilp_sol(tdag, vp, use_pulp=True)
    if 'maino_v' in io_budgets:
        v_main_o = io_budgets['maino_v']
        mo_lvl = assign.v_lvl_in[v_main_o]
        mo_scl = assign.v_scl_in[v_main_o]
        assert (mo_lvl, mo_scl) in io_budgets['main_qbp_cost'], f"Main output (lvl, scl)=({mo_lvl}, {mo_scl}) not in provided QBP costs"

    assign_cost = estimate_assign(assign, le)
    return assign, assign_cost
