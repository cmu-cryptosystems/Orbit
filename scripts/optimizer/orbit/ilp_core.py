from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

import gurobipy as gp
from gurobipy import GRB

class VarPool:
    def __init__(self, tdag: Tdag, params: Params, model: gp.Model):
        self.params = params
        self.Smax = params.Sf + 2 * params.Sw
        self.model = model
        
        self.vars = dict()
        self.total_cost = []
        self.error_penalty_terms = []
        self.error_enabled = (
            params.resilience_mode == "error-state"
            and params.resilience_profile is not None
        )
        
        # Node variables, skip constant nodes
        for v in tdag.nodes:
            if tdag.nodes[v]['op'] == 'constant':
                continue
            self.vars[f"v_lvl_in_{v}"] = model.addVar(
                lb=1, ub=params.lvl_ub, vtype=GRB.INTEGER, name=f"v_lvl_in_{v}"
            )
            self.vars[f"v_scl_in_{v}"] = model.addVar(
                lb=self._node_scale_lb(tdag, v, "in"),
                ub=self.Smax,
                vtype=GRB.INTEGER,
                name=f"v_scl_in_{v}",
            )
            self.vars[f"v_lvl_out_{v}"] = model.addVar(
                lb=1, ub=params.lvl_ub, vtype=GRB.INTEGER, name=f"v_lvl_out_{v}"
            )
            self.vars[f"v_scl_out_{v}"] = model.addVar(
                lb=self._node_scale_lb(tdag, v, "out"),
                ub=self.Smax,
                vtype=GRB.INTEGER,
                name=f"v_scl_out_{v}",
            )
            self.vars[f"v_use_r_{v}"] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_use_r_{v}")
            self.vars[f"v_use_b_{v}"] = model.addVar(vtype=GRB.BINARY, name=f"v_use_b_{v}")
            if self.error_enabled:
                max_error = params.resilience_error_model["max_error_abs"]
                self.vars[f"v_err_in_{v}"] = model.addVar(
                    lb=0.0,
                    ub=max_error,
                    vtype=GRB.CONTINUOUS,
                    name=f"v_err_in_{v}",
                )
                self.vars[f"v_err_out_{v}"] = model.addVar(
                    lb=0.0,
                    ub=max_error,
                    vtype=GRB.CONTINUOUS,
                    name=f"v_err_out_{v}",
                )
                self.error_penalty_terms.append(self.vars[f"v_err_in_{v}"])
                self.error_penalty_terms.append(self.vars[f"v_err_out_{v}"])
        
        self.rescale_cost = None
        
        # Edge variables, skip constant->node edges
        for u, v in tdag.edges:
            if tdag.nodes[u]['op'] == 'constant':
                continue
            edge_label = self.get_edge_label(u, v)
            self.vars[f"e_lvl_in_{edge_label}"] = self.vars[f"v_lvl_out_{u}"]
            self.vars[f"e_scl_in_{edge_label}"] = self.vars[f"v_scl_out_{u}"]
            self.vars[f"e_lvl_out_{edge_label}"] = self.vars[f"v_lvl_in_{v}"]
            if self.error_enabled:
                max_error = params.resilience_error_model["max_error_abs"]
                self.vars[f"e_err_in_{edge_label}"] = self.vars[f"v_err_out_{u}"]
                self.vars[f"e_err_out_{edge_label}"] = model.addVar(
                    lb=0.0,
                    ub=max_error,
                    vtype=GRB.CONTINUOUS,
                    name=f"e_err_out_{edge_label}",
                )
                self.error_penalty_terms.append(self.vars[f"e_err_out_{edge_label}"])
            if tdag.nodes[v]['op'] == 'mul':
                self.vars[f"e_scl_out_{edge_label}"] = model.addVar(
                    lb=self._edge_scale_lb(tdag, u, v),
                    ub=self.Smax,
                    vtype=GRB.INTEGER,
                    name=f"e_scl_out_{edge_label}",
                )
            else:
                self.vars[f"e_scl_out_{edge_label}"] = self.vars[f"v_scl_in_{v}"]
                model.addConstr(
                    self.vars[f"e_scl_out_{edge_label}"] >= self._edge_scale_lb(tdag, u, v),
                    name=f"edge_scale_lb_{edge_label}",
                )
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

    def _node_scale_lb(self, tdag: Tdag, v: str, port: str) -> int:
        scale_lb = self.params.scale_lower_bound(v, tdag.nodes[v], port)
        if scale_lb > self.Smax:
            raise ValueError(
                f"Node {v} has resilience min_scale={scale_lb}, "
                f"above Orbit Smax={self.Smax}."
            )
        return scale_lb

    def _edge_scale_lb(self, tdag: Tdag, u: str, v: str) -> int:
        return max(
            self._node_scale_lb(tdag, u, "out"),
            self._node_scale_lb(tdag, v, "in"),
        )

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

    def var_err(self, v: str | tuple[str, str], type: str) -> gp.Var:
        if isinstance(v, str):
            return self.vars[f"v_err_{type}_{v}"]
        edge_label = self.get_edge_label(v[0], v[1])
        return self.vars[f"e_err_{type}_{edge_label}"]

def add_ilp_constraints(tdag: Tdag, vp: VarPool):
    model = vp.model
    params = vp.params
    def decryptable_scale_bound(level_var):
        return params.Sf * (level_var - params.lvl_lb + 2) - 7

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
        model.addConstr(vp.var_scl(v,'in') == vp.var_scl((i0,v), 'out') + vp.var_scl((i1,v), 'out'), name=f"scl_mul_{v}_in_{i0}_{i1}_eq")
    # add node rescale/bootstrap constraints
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
        model.addConstr(vp.var_scl(v, 'in') <= decryptable_scale_bound(vp.var_lvl(v, 'in')),
                        name=f"decryptable_{v}_input_scale")
        model.addConstr(vp.var_scl(v, 'out') <= decryptable_scale_bound(vp.var_lvl(v, 'out')),
                        name=f"decryptable_{v}_output_scale")
        model.addGenConstrIndicator(vp.var_use(v, 'b'), True,
                                    vp.var_scl(v,'in') <= params.Sf * (vp.var_lvl(v,'in')-params.bts_lb+1),
                                    name=f"bts_{v}_input_lvl_scl")
        model.addGenConstrIndicator(vp.var_use(v, 'b'), True,
                                    vp.var_lvl(v,'out') >= params.bts_lb + 1,
                                    name=f"bts_{v}_output_lvl")
        model.addGenConstrIndicator(vp.var_use(v, 'b'), True,
                                    vp.var_scl(v,'out') >= params.Sf,
                                    name=f"bts_{v}_output_scl")
        # add not using bootstrapping constraints
        model.addGenConstrIndicator(vp.var_use(v, 'b'), False,
                                    vp.var_lvl(v,'out') <= vp.var_lvl(v,'in') - vp.var_use(v,'r'),
                                    name=f"nobts_{v}_lvl")
        model.addGenConstrIndicator(vp.var_use(v, 'b'), False,
                                    vp.var_scl(v,'out') >= vp.var_scl(v,'in') - params.Sf * vp.var_use(v,'r'),
                                    name=f"nobts_{v}_scl")
    # add edge rescale constraints
    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        edge_label = vp.get_edge_label(u, v)
        model.addConstr(vp.var_lvl((u,v),'out') <= vp.var_lvl((u,v),'in') - vp.var_use((u,v),'r'),
                        name=f"edge_rescale_{edge_label}_lvl")
        model.addConstr(vp.var_scl((u,v),'out') >= vp.var_scl((u,v),'in') - params.Sf * vp.var_use((u,v),'r'),
                        name=f"edge_rescale_{edge_label}_scl")

def _node_compute_error(tdag: Tdag, v: str, params: Params) -> float:
    model = params.resilience_error_model
    op = tdag.nodes[v]['op']
    if op == 'input':
        return model["input_error_abs"]
    if op == 'add':
        single, double = tdag.get_v_weights(v)
        return model["add_error_abs"] * max(1, single + double)
    if op == 'mul':
        descr = tdag.nodes[v]['op_descr']
        if descr.get('double', 0) > 0:
            return model["mul_cipher_error_abs"]
        return model["mul_plain_error_abs"]
    if op == 'rotate':
        return model["rotate_error_abs"]
    return model["add_error_abs"]

def add_error_state_constraints(tdag: Tdag, vp: VarPool, io_budgets: dict | None = None):
    if not vp.error_enabled:
        return

    model = vp.model
    params = vp.params
    error_model = params.resilience_error_model
    io_budgets = io_budgets or {}
    external_input_error = io_budgets.get("in_error_abs")

    # Edge maintenance currently has explicit rescale variables. If an edge
    # later decodes to a bootstrap via level feasibility, this conservative
    # recurrence overestimates error because it does not model an edge reset.
    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        model.addConstr(
            vp.var_err((u, v), 'out')
            == vp.var_err((u, v), 'in')
            + error_model["rescale_error_abs"] * vp.var_use((u, v), 'r'),
            name=f"edge_error_{vp.get_edge_label(u, v)}",
        )

    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue

        pred_errors = [
            vp.var_err((u, v), 'out')
            for u in tdag.predecessors(v)
            if tdag.nodes[u]['op'] != 'constant'
        ]
        if tdag.nodes[v]['op'] == 'input':
            # SISO partitioning turns each window boundary into an input node.
            # The bounded-DP resilience search passes the previous window's
            # output error here so tau constraints see cumulative error instead
            # of restarting at zero for every micro-partition.
            if external_input_error is not None and v in tdag.inputs:
                compute_error = float(external_input_error)
            else:
                compute_error = error_model["input_error_abs"]
        else:
            compute_error = _node_compute_error(tdag, v, params)
        model.addConstr(
            vp.var_err(v, 'in') == gp.quicksum(pred_errors) + compute_error,
            name=f"node_compute_error_{v}",
        )

        max_error = error_model["max_error_abs"]
        no_bootstrap_error = (
            vp.var_err(v, 'in')
            + error_model["rescale_error_abs"] * vp.var_use(v, 'r')
        )
        model.addConstr(
            vp.var_err(v, 'out')
            >= error_model["bootstrap_error_abs"] - max_error * (1 - vp.var_use(v, 'b')),
            name=f"node_bootstrap_error_reset_lb_{v}",
        )
        model.addConstr(
            vp.var_err(v, 'out')
            <= error_model["bootstrap_error_abs"] + max_error * (1 - vp.var_use(v, 'b')),
            name=f"node_bootstrap_error_reset_ub_{v}",
        )
        model.addConstr(
            vp.var_err(v, 'out') >= no_bootstrap_error - max_error * vp.var_use(v, 'b'),
            name=f"node_no_bootstrap_error_lb_{v}",
        )
        model.addConstr(
            vp.var_err(v, 'out') <= no_bootstrap_error + max_error * vp.var_use(v, 'b'),
            name=f"node_no_bootstrap_error_ub_{v}",
        )

        if params.resilience_constraint_policy == "hard-tau":
            tau_in = params.resilience_profile.error_upper_bound(v, tdag.nodes[v], "in")
            if tau_in is not None:
                model.addConstr(vp.var_err(v, 'in') <= tau_in, name=f"tau_in_{v}")
            tau_out = params.resilience_profile.error_upper_bound(v, tdag.nodes[v], "out")
            if tau_out is not None:
                model.addConstr(vp.var_err(v, 'out') <= tau_out, name=f"tau_out_{v}")

def add_ilp_linear_cost(tdag: Tdag, vp: VarPool, le: LatencyEstimator):
    cost_rescale_s = le.lin_op_lmaps['rescale_single']
    cost_bts_s = le.lin_op_lmaps['bootstrap_single']
    vp.rescale_cost = cost_rescale_s[1]
    
    # add node rescale/bootstrap costs
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
        vp.total_cost.append(tdag.nodes[v]['weight'] * cost_bts_s[1] * vp.var_use(v, 'b'))
        vp.total_cost.append(tdag.nodes[v]['weight'] * cost_rescale_s[1] * vp.var_use(v,'r'))
    # add edge rescale costs
    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        vp.total_cost.append(tdag.edges[u, v]['weight'] * cost_rescale_s[1] * vp.var_use((u,v),'r'))
    for v in tdag.nodes:
        single_cnt, double_cnt = tdag.get_v_weights(v)
        if single_cnt > 0:
            this_cost = le.lin_op_lmaps[tdag.nodes[v]['op'] + '_single']
            vp.total_cost.append(single_cnt * (this_cost[0] * vp.var_lvl(v, 'in') + this_cost[1]))
        if double_cnt > 0:
            this_cost = le.lin_op_lmaps[tdag.nodes[v]['op'] + '_double']
            vp.total_cost.append(double_cnt * (this_cost[0] * vp.var_lvl(v, 'in') + this_cost[1]))
    if vp.error_enabled:
        # Tiny tie-breaker: keep unconstrained error variables at their minimum
        # feasible value without materially changing latency optimization.
        vp.total_cost.append(1e-6 * gp.quicksum(vp.error_penalty_terms))

def add_ilp_io_budgets(tdag: Tdag, vp: VarPool, io_budgets):
    v_in = list(tdag.inputs)[0]
    v_out = list(tdag.outputs)[0]
    model = vp.model
    params = vp.params
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
    model = vp.model
    model.setParam('OutputFlag', 0)
    model.setParam("Seed", 42)
    model.setParam('Method', 2) 
    model.setParam('Threads', num_threads)
    model.setParam('MIPGap', 0.01)
    time_limit = vp.params.ilp_task_time_limit_sec
    model.setParam('TimeLimit', time_limit if time_limit and time_limit > 0 else GRB.INFINITY)
    
    model.setObjective(gp.quicksum(vp.total_cost), GRB.MINIMIZE)
    model.optimize()
    
def solve_ilp_core_bypass(tdag: Tdag, vp: VarPool, io_budgets, num_threads: int):
    model = vp.model
    params = vp.params
    model.setParam('OutputFlag', 0)
    model.setParam("Seed", 42)
    model.setParam('Method', 2) 
    model.setParam('Threads', num_threads)
    model.setParam('MIPGap', 0.01)
    time_limit = params.ilp_task_time_limit_sec
    model.setParam('TimeLimit', time_limit if time_limit and time_limit > 0 else GRB.INFINITY)
    
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

def solve_ilp(tdag: Tdag, io_budgets: dict, le: LatencyEstimator, task_name: str, num_threads: int, params: Params
              ) -> tuple[Assign|None, float|None]:
    model = gp.Model(task_name)
    vp = VarPool(tdag, params, model)
    
    add_ilp_constraints(tdag, vp)
    add_error_state_constraints(tdag, vp, io_budgets)
    add_ilp_linear_cost(tdag, vp, le)
    add_ilp_io_budgets(tdag, vp, io_budgets)
    if 'maino_v' in io_budgets:
        solve_ilp_core_bypass(tdag, vp, io_budgets, num_threads)
    else:
        solve_ilp_core(vp, num_threads)
    
    if model.SolCount == 0:
        return None, None
    assign = decode_ilp_sol(tdag, vp)
    if 'maino_v' in io_budgets:
        v_main_o = io_budgets['maino_v']
        mo_lvl = assign.v_lvl_in[v_main_o]
        mo_scl = assign.v_scl_in[v_main_o]
        assert (mo_lvl, mo_scl) in io_budgets['main_qbp_cost'], f"Main output (lvl, scl)=({mo_lvl}, {mo_scl}) not in provided QBP costs"
    
    assign_cost = estimate_assign(assign, le)
    return assign, assign_cost

def decode_ilp_sol(tdag: Tdag, vp: VarPool) -> Assign:
    assign = Assign(tdag)
    params = vp.params
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'input':
            # input nodes, need to store in-level/scale
            assign.v_lvl_in[v] = round(vp.var_lvl(v, 'in').X)
            assign.v_scl_in[v] = round(vp.var_scl(v, 'in').X)
            input_scale_lb = params.scale_lower_bound(v, tdag.nodes[v], "in")
            assert assign.v_scl_in[v] >= input_scale_lb, (
                f"Node {v} input scale {assign.v_scl_in[v]} below "
                f"local lower bound {input_scale_lb}"
            )
        if vp.error_enabled and tdag.nodes[v]['op'] != 'constant':
            assign.v_err_in[v] = float(vp.var_err(v, 'in').X)
            assign.v_err_out[v] = float(vp.var_err(v, 'out').X)
        assign.v_lvl_out[v] = round(vp.var_lvl(v, 'out').X)
        assign.v_scl_out[v] = round(vp.var_scl(v, 'out').X)
        if tdag.nodes[v]['op'] != 'constant':
            output_scale_lb = params.scale_lower_bound(v, tdag.nodes[v], "out")
            assert assign.v_scl_out[v] >= output_scale_lb, (
                f"Node {v} output scale {assign.v_scl_out[v]} below "
                f"local lower bound {output_scale_lb}"
            )
    
    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        assign.e_lvl_out[(u,v)] = round(vp.var_lvl((u,v), 'out').X)
        assign.e_scl_out[(u,v)] = round(vp.var_scl((u,v), 'out').X)
        if vp.error_enabled:
            assign.e_err_out[(u, v)] = float(vp.var_err((u, v), 'out').X)
        edge_scale_lb = max(
            params.scale_lower_bound(u, tdag.nodes[u], "out"),
            params.scale_lower_bound(v, tdag.nodes[v], "in"),
        )
        assert assign.e_scl_out[(u,v)] >= edge_scale_lb, (
            f"Edge ({u},{v}) output scale {assign.e_scl_out[(u,v)]} below "
            f"local lower bound {edge_scale_lb}"
        )
    
    return assign
