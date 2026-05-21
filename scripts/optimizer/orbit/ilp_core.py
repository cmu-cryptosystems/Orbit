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
        
        # Node variables, skip constant nodes
        for v in tdag.nodes:
            if tdag.nodes[v]['op'] == 'constant':
                continue
            self.vars[f"v_lvl_in_{v}"] = model.addVar(lb=1, ub=params.lvl_ub, vtype=GRB.INTEGER, name=f"v_lvl_in_{v}")
            self.vars[f"v_scl_in_{v}"] = model.addVar(
                lb=self._node_scale_lb(tdag, v, "in"),
                ub=self.Smax,
                vtype=GRB.INTEGER,
                name=f"v_scl_in_{v}",
            )
            self.vars[f"v_lvl_out_{v}"] = model.addVar(lb=1, ub=params.lvl_ub, vtype=GRB.INTEGER, name=f"v_lvl_out_{v}")
            self.vars[f"v_scl_out_{v}"] = model.addVar(
                lb=self._node_scale_lb(tdag, v, "out"),
                ub=self.Smax,
                vtype=GRB.INTEGER,
                name=f"v_scl_out_{v}",
            )
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
                const_scale = params.constant_scale_for_node(v, tdag.nodes[v])
                self.vars[f"v_scl_out_{u}"] = model.addVar(lb=const_scale, ub=const_scale, vtype=GRB.INTEGER, name=f"v_scl_out_{u}")
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
                f"Node {v} has local min_scale={scale_lb}, above Orbit Smax={self.Smax}."
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
        model.addConstr(vp.var_scl(v,'in') == vp.var_scl((i0,v), 'out') + vp.var_scl((i1,v), 'out'), name=f"scl_mul_{v}_in_{i0}_{i1}_eq")
    # add node rescale/bootstrap constraints
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
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

def add_ilp_upscale_proxy_cost(tdag: Tdag, vp: VarPool, le: LatencyEstimator):
    params = vp.params
    weight = getattr(params, "upscale_objective_weight", 0.0)
    if weight <= 0:
        return

    upscale_costs = le.op_lmaps.get('upscale_single', {})
    if not upscale_costs:
        return
    avg_upscale_cost = sum(upscale_costs.values()) / len(upscale_costs)
    cost_per_scale_bit = weight * avg_upscale_cost / params.Sf
    model = vp.model

    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
        deficit = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"v_upscale_deficit_{v}")
        vp.vars[f"v_upscale_deficit_{v}"] = deficit
        model.addConstr(
            deficit >= vp.var_scl(v, 'out') + params.Sf * vp.var_use(v, 'r') - vp.var_scl(v, 'in'),
            name=f"v_upscale_deficit_{v}_lb",
        )
        vp.total_cost.append(tdag.nodes[v]['weight'] * cost_per_scale_bit * deficit)

    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        edge_label = vp.get_edge_label(u, v)
        deficit = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"e_upscale_deficit_{edge_label}")
        vp.vars[f"e_upscale_deficit_{edge_label}"] = deficit
        model.addConstr(
            deficit >= vp.var_scl((u, v), 'out') + params.Sf * vp.var_use((u, v), 'r') - vp.var_scl((u, v), 'in'),
            name=f"e_upscale_deficit_{edge_label}_lb",
        )
        vp.total_cost.append(tdag.edges[u, v]['weight'] * cost_per_scale_bit * deficit)

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
    model.setParam('TimeLimit', GRB.INFINITY)
    
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

def solve_ilp(tdag: Tdag, io_budgets: dict, le: LatencyEstimator, task_name: str, num_threads: int, params: Params
              ) -> tuple[Assign|None, float|None]:
    model = gp.Model(task_name)
    vp = VarPool(tdag, params, model)
    
    add_ilp_constraints(tdag, vp)
    add_ilp_linear_cost(tdag, vp, le)
    add_ilp_upscale_proxy_cost(tdag, vp, le)
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
            assert assign.v_scl_in[v] >= input_scale_lb, f"Node {v} input scale {assign.v_scl_in[v]} below local lower bound {input_scale_lb}"
        assign.v_lvl_out[v] = round(vp.var_lvl(v, 'out').X)
        assign.v_scl_out[v] = round(vp.var_scl(v, 'out').X)
        if tdag.nodes[v]['op'] != 'constant':
            output_scale_lb = params.scale_lower_bound(v, tdag.nodes[v], "out")
            assert assign.v_scl_out[v] >= output_scale_lb, f"Node {v} output scale {assign.v_scl_out[v]} below local lower bound {output_scale_lb}"
    
    for u, v in tdag.edges:
        if tdag.nodes[u]['op'] == 'constant':
            continue
        assign.e_lvl_out[(u,v)] = round(vp.var_lvl((u,v), 'out').X)
        assign.e_scl_out[(u,v)] = round(vp.var_scl((u,v), 'out').X)
        edge_scale_lb = max(
            params.scale_lower_bound(u, tdag.nodes[u], "out"),
            params.scale_lower_bound(v, tdag.nodes[v], "in"),
        )
        assert assign.e_scl_out[(u,v)] >= edge_scale_lb, f"Edge ({u},{v}) output scale {assign.e_scl_out[(u,v)]} below local lower bound {edge_scale_lb}"
    
    return assign
