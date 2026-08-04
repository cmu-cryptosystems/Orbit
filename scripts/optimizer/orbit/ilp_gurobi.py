"""Gurobi-based ILP building blocks (optional; requires gurobipy)."""
from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from ...tdag import Tdag
from ...params.params import Params


class VarPool:
    def __init__(self, tdag: Tdag, params: Params, model: gp.Model):
        self.params = params
        self.Smax = params.Sf + 2 * params.Sw
        self.model = model

        self.vars = dict()
        self.total_cost = []

        # Node variables, skip constant nodes
        for v in tdag.nodes:
            if tdag.nodes[v]['op'] == 'constant':
                continue
            self.vars[f"v_lvl_in_{v}"] = model.addVar(lb=1, ub=params.lvl_ub, vtype=GRB.INTEGER, name=f"v_lvl_in_{v}")
            self.vars[f"v_scl_in_{v}"] = model.addVar(lb=params.Sw, ub=self.Smax, vtype=GRB.INTEGER, name=f"v_scl_in_{v}")
            self.vars[f"v_lvl_out_{v}"] = model.addVar(lb=1, ub=params.lvl_ub, vtype=GRB.INTEGER, name=f"v_lvl_out_{v}")
            self.vars[f"v_scl_out_{v}"] = model.addVar(lb=params.Sw, ub=self.Smax, vtype=GRB.INTEGER, name=f"v_scl_out_{v}")
            self.vars[f"v_use_r_{v}"] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_use_r_{v}")
            self.vars[f"v_use_b_{v}"] = model.addVar(vtype=GRB.BINARY, name=f"v_use_b_{v}")

        self.rescale_cost = None

        # Edge variables, skip constant->node edges
        for u, v in tdag.edges:
            if tdag.nodes[u]['op'] == 'constant':
                continue
            edge_label = self.get_edge_label(u, v)
            self.vars[f"e_lvl_in_{edge_label}"] = self.vars[f"v_lvl_out_{u}"]
            self.vars[f"e_scl_in_{edge_label}"] = self.vars[f"v_scl_out_{u}"]
            self.vars[f"e_lvl_out_{edge_label}"] = self.vars[f"v_lvl_in_{v}"]
            if tdag.nodes[v]['op'] == 'mul':
                self.vars[f"e_scl_out_{edge_label}"] = model.addVar(lb=params.Sw, ub=self.Smax, vtype=GRB.INTEGER, name=f"e_scl_out_{edge_label}")
            else:
                self.vars[f"e_scl_out_{edge_label}"] = self.vars[f"v_scl_in_{v}"]
            self.vars[f"e_use_r_{edge_label}"] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"e_use_r_{edge_label}")

        # Constant variables
        for u in tdag.nodes:
            if tdag.nodes[u]['op'] != 'constant':
                continue
            assert tdag.out_degree(u) == 1, f"Constant node {u} should have exactly one output"
            v = list(tdag.successors(u))[0]
            # node variables
            self.vars[f"v_lvl_out_{u}"] = self.vars[f"v_lvl_in_{v}"]
            if tdag.nodes[v]['op'] == 'mul':
                self.vars[f"v_scl_out_{u}"] = model.addVar(lb=params.Csw, ub=params.Csw, vtype=GRB.INTEGER, name=f"v_scl_out_{u}")
            else:
                self.vars[f"v_scl_out_{u}"] = self.vars[f"v_scl_in_{v}"]
            # edge variables
            edge_label = self.get_edge_label(u, v)
            self.vars[f"e_lvl_out_{edge_label}"] = self.vars[f"v_lvl_out_{u}"]
            self.vars[f"e_scl_out_{edge_label}"] = self.vars[f"v_scl_out_{u}"]

    def get_edge_label(self, u: str, v: str) -> str:
        return f"({u}_{v})"

    def var_lvl(self, v: str | tuple[str, str], type: str) -> gp.Var:
        if isinstance(v, str):
            return self.vars[f"v_lvl_{type}_{v}"]
        else:
            edge_label = self.get_edge_label(v[0], v[1])
            return self.vars[f"e_lvl_{type}_{edge_label}"]

    def var_scl(self, v: str | tuple[str, str], type: str) -> gp.Var:
        if isinstance(v, str):
            return self.vars[f"v_scl_{type}_{v}"]
        else:
            edge_label = self.get_edge_label(v[0], v[1])
            return self.vars[f"e_scl_{type}_{edge_label}"]

    def var_use(self, v: str | tuple[str, str], type: str) -> gp.Var:
        if isinstance(v, str):
            return self.vars[f"v_use_{type}_{v}"]
        else:
            edge_label = self.get_edge_label(v[0], v[1])
            return self.vars[f"e_use_{type}_{edge_label}"]


def add_ilp_constraints(tdag: Tdag, vp: VarPool):
    model = vp.model
    params = vp.params
    # add mul scale constraints
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] != 'mul':
            continue
        assert 1 <= tdag.in_degree(v) <= 2, f"Mul node {v} should have 1 or 2 inputs"
        ilist = list(tdag.predecessors(v))
        if len(ilist) == 1:
            i0, i1 = ilist[0], ilist[0]
        else:
            i0, i1 = ilist[0], ilist[1]
        model.addConstr(vp.var_scl(v, 'in') == vp.var_scl((i0, v), 'out') + vp.var_scl((i1, v), 'out'), name=f"scl_mul_{v}_in_{i0}_{i1}_eq")
    # add node rescale/bootstrap constraints
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
        model.addGenConstrIndicator(vp.var_use(v, 'b'), True,
                                    vp.var_scl(v, 'in') <= params.Sf * (vp.var_lvl(v, 'in') - params.bts_lb + 1),
                                    name=f"bts_{v}_input_lvl_scl")
        model.addGenConstrIndicator(vp.var_use(v, 'b'), True,
                                    vp.var_lvl(v, 'out') >= params.bts_lb + 1,
                                    name=f"bts_{v}_output_lvl")
        model.addGenConstrIndicator(vp.var_use(v, 'b'), True,
                                    vp.var_scl(v, 'out') >= params.Sf,
                                    name=f"bts_{v}_output_scl")
        # add not using bootstrapping constraints
        model.addGenConstrIndicator(vp.var_use(v, 'b'), False,
                                    vp.var_lvl(v, 'out') <= vp.var_lvl(v, 'in') - vp.var_use(v, 'r'),
                                    name=f"nobts_{v}_lvl")
        model.addGenConstrIndicator(vp.var_use(v, 'b'), False,
                                    vp.var_scl(v, 'out') >= vp.var_scl(v, 'in') - params.Sf * vp.var_use(v, 'r'),
                                    name=f"nobts_{v}_scl")
    # add edge rescale constraints
    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        edge_label = vp.get_edge_label(u, v)
        model.addConstr(vp.var_lvl((u, v), 'out') <= vp.var_lvl((u, v), 'in') - vp.var_use((u, v), 'r'),
                        name=f"edge_rescale_{edge_label}_lvl")
        model.addConstr(vp.var_scl((u, v), 'out') >= vp.var_scl((u, v), 'in') - params.Sf * vp.var_use((u, v), 'r'),
                        name=f"edge_rescale_{edge_label}_scl")


def add_ilp_io_budgets(tdag: Tdag, vp: VarPool, io_budgets):
    model = vp.model
    params = vp.params
    v_in = list(tdag.inputs)[0]
    v_out = list(tdag.outputs)[0]
    if 'in_lvl' in io_budgets and io_budgets['in_lvl'] >= 0:
        in_lvl = io_budgets['in_lvl']
        model.addConstr(vp.var_lvl(v_in, 'in') == in_lvl, name=f"input_level_{in_lvl}")
    if 'in_scl' in io_budgets and io_budgets['in_scl'] >= 0:
        in_scl = io_budgets['in_scl']
        model.addConstr(vp.var_scl(v_in, 'in') == in_scl, name=f"input_scale_{in_scl}")
    if 'out_lvl' in io_budgets and io_budgets['out_lvl'] >= 0:
        out_lvl = io_budgets['out_lvl']
        model.addConstr(vp.var_lvl(v_out, 'out') == out_lvl, name=f"output_level_{out_lvl}")

    model.addConstr(vp.var_scl(v_out, 'out') <= params.Sf * (vp.var_lvl(v_out, 'out') + 1) - 7,
                    name=f"lattigo_output_scale_level_relation")

    vp.total_cost.append(0.2 * vp.rescale_cost * tdag.get_full_size() * vp.var_scl(v_out, 'out'))


def solve_ilp_core(vp: VarPool, num_threads: int):
    """Solve the standard (non-bypass) partition ILP.

    Sets Orbit's fixed Gurobi parameters (deterministic ``Seed=42``, barrier
    ``Method=2``, ``MIPGap=0.01``, ``Threads=num_threads``) and minimizes the
    accumulated rescale/bootstrap objective.
    """
    model = vp.model
    model.setParam('OutputFlag', 0)
    model.setParam("Seed", 42)
    model.setParam('Method', 2)
    model.setParam('Threads', num_threads)
    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', GRB.INFINITY)

    model.setObjective(gp.quicksum(vp.total_cost), GRB.MINIMIZE)
    model.optimize()


def solve_ilp_core_bypass(tdag: Tdag, vp: VarPool, io_budgets, num_threads: int):
    """Solve the bypass-aware partition ILP.

    For each candidate main-output (level, scale) in
    ``io_budgets['main_qbp_cost']`` the model is re-optimized with that pair
    pinned; the choice minimizing the combined main + bypass + fresh-tensor cost
    is kept and re-solved as the final assignment. Returns early (leaving the
    model unsolved) if no pinned configuration is feasible.
    """
    model = vp.model
    model.setParam('OutputFlag', 0)
    model.setParam("Seed", 42)
    model.setParam('Method', 2)
    model.setParam('Threads', num_threads)
    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', GRB.INFINITY)

    model.setObjective(gp.quicksum(vp.total_cost), GRB.MINIMIZE)

    v_main_o = io_budgets['maino_v']
    assert v_main_o in tdag.nodes, f"Main output node {v_main_o} not in Tdag {tdag.name}"
    main_dag_size = io_budgets['main_dag_size']
    main_qbp_cost = io_budgets['main_qbp_cost']
    all_out = list(tdag.outputs)[0]

    # try all possible main-out (lvl, scl) configurations
    min_cost = None
    min_cost_ls = None

    def add_cons(mo_lvl, mo_scl):
        lvl_constr = model.addConstr(vp.var_lvl(v_main_o, 'in') == mo_lvl, name=f"main_output_level_{mo_lvl}")
        scl_constr = model.addConstr(vp.var_scl(v_main_o, 'in') == mo_scl, name=f"main_output_scale_{mo_scl}")
        return lvl_constr, scl_constr

    for (mo_lvl, mo_scl), main_cost in main_qbp_cost.items():
        tmp_cons_list = add_cons(mo_lvl, mo_scl)
        model.optimize()
        if model.SolCount != 0:
            ft_cost = 0.2 * vp.rescale_cost * main_dag_size * round(vp.var_scl(all_out, 'out').X)
            this_cost = model.ObjVal + main_cost + ft_cost
            if (min_cost is None) or (this_cost < min_cost):
                min_cost = this_cost
                min_cost_ls = (mo_lvl, mo_scl)
        model.remove(tmp_cons_list)
        model.update()

    if min_cost_ls is None:
        return
    add_cons(min_cost_ls[0], min_cost_ls[1])
    model.optimize()


def gurobi_has_solution(model: gp.Model) -> bool:
    return model.SolCount != 0
