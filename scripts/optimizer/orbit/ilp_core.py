from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

import pulp


class VarPool:
    def __init__(self, tdag: Tdag, params: Params, model: pulp.LpProblem):
        self.params = params
        self.Smax = params.Sf + 2 * params.Sw
        self.model = model

        self.vars = dict()
        self.total_cost = []

        # Node variables, skip constant nodes
        for v in tdag.nodes:
            if tdag.nodes[v]["op"] == "constant":
                continue
            self.vars[f"v_lvl_in_{v}"] = pulp.LpVariable(
                f"v_lvl_in_{v}",
                lowBound=1,
                upBound=params.lvl_ub,
                cat=pulp.LpInteger,
            )
            self.vars[f"v_scl_in_{v}"] = pulp.LpVariable(
                f"v_scl_in_{v}",
                lowBound=params.Sw,
                upBound=self.Smax,
                cat=pulp.LpInteger,
            )
            self.vars[f"v_lvl_out_{v}"] = pulp.LpVariable(
                f"v_lvl_out_{v}",
                lowBound=1,
                upBound=params.lvl_ub,
                cat=pulp.LpInteger,
            )
            self.vars[f"v_scl_out_{v}"] = pulp.LpVariable(
                f"v_scl_out_{v}",
                lowBound=params.Sw,
                upBound=self.Smax,
                cat=pulp.LpInteger,
            )
            self.vars[f"v_use_r_{v}"] = pulp.LpVariable(
                f"v_use_r_{v}",
                lowBound=0,
                cat=pulp.LpInteger,
            )
            self.vars[f"v_use_b_{v}"] = pulp.LpVariable(
                f"v_use_b_{v}",
                cat=pulp.LpBinary,
            )

        self.rescale_cost = None

        # Edge variables, skip constant->node edges
        for u, v in tdag.edges:
            if tdag.nodes[u]["op"] == "constant":
                continue
            edge_label = self.get_edge_label(u, v)
            self.vars[f"e_lvl_in_{edge_label}"] = self.vars[f"v_lvl_out_{u}"]
            self.vars[f"e_scl_in_{edge_label}"] = self.vars[f"v_scl_out_{u}"]
            self.vars[f"e_lvl_out_{edge_label}"] = self.vars[f"v_lvl_in_{v}"]
            if tdag.nodes[v]["op"] == "mul":
                self.vars[f"e_scl_out_{edge_label}"] = pulp.LpVariable(
                    f"e_scl_out_{edge_label}",
                    lowBound=params.Sw,
                    upBound=self.Smax,
                    cat=pulp.LpInteger,
                )
            else:
                self.vars[f"e_scl_out_{edge_label}"] = self.vars[f"v_scl_in_{v}"]
            self.vars[f"e_use_r_{edge_label}"] = pulp.LpVariable(
                f"e_use_r_{edge_label}",
                lowBound=0,
                cat=pulp.LpInteger,
            )

        # Constant variables
        for u in tdag.nodes:
            if tdag.nodes[u]["op"] != "constant":
                continue
            assert tdag.out_degree(u) == 1, (
                f"Constant node {u} should have exactly one output"
            )
            v = list(tdag.successors(u))[0]
            # node variables
            self.vars[f"v_lvl_out_{u}"] = self.vars[f"v_lvl_in_{v}"]
            if tdag.nodes[v]["op"] == "mul":
                self.vars[f"v_scl_out_{u}"] = pulp.LpVariable(
                    f"v_scl_out_{u}",
                    lowBound=params.Csw,
                    upBound=params.Csw,
                    cat=pulp.LpInteger,
                )
            else:
                self.vars[f"v_scl_out_{u}"] = self.vars[f"v_scl_in_{v}"]
            # edge variables
            edge_label = self.get_edge_label(u, v)
            self.vars[f"e_lvl_out_{edge_label}"] = self.vars[f"v_lvl_out_{u}"]
            self.vars[f"e_scl_out_{edge_label}"] = self.vars[f"v_scl_out_{u}"]

    def get_edge_label(self, u: str, v: str) -> str:
        return f"({u}_{v})"

    def var_lvl(self, v: str | tuple[str, str], type: str):
        if isinstance(v, str):
            return self.vars[f"v_lvl_{type}_{v}"]
        else:
            edge_label = self.get_edge_label(v[0], v[1])
            return self.vars[f"e_lvl_{type}_{edge_label}"]

    def var_scl(self, v: str | tuple[str, str], type: str):
        if isinstance(v, str):
            return self.vars[f"v_scl_{type}_{v}"]
        else:
            edge_label = self.get_edge_label(v[0], v[1])
            return self.vars[f"e_scl_{type}_{edge_label}"]

    def var_use(self, v: str | tuple[str, str], type: str):
        if isinstance(v, str):
            return self.vars[f"v_use_{type}_{v}"]
        else:
            edge_label = self.get_edge_label(v[0], v[1])
            return self.vars[f"e_use_{type}_{edge_label}"]


def add_ilp_constraints(tdag: Tdag, vp: VarPool):
    model = vp.model
    params = vp.params
    big_m_level = params.lvl_ub + params.bts_ub + 2
    big_m_scale = 2 * vp.Smax + params.Sf * (params.lvl_ub + params.bts_ub + 2)

    # add mul scale constraints
    for v in tdag.nodes:
        if tdag.nodes[v]["op"] != "mul":
            continue
        assert 1 <= tdag.in_degree(v) <= 2, (
            f"Mul node {v} should have 1 or 2 inputs"
        )
        ilist = list(tdag.predecessors(v))
        if len(ilist) == 1:
            i0 = ilist[0]
            if tdag.nodes[v]["op_descr"].get("single", 0) > 0 and tdag.nodes[v]["op_descr"].get("double", 0) == 0:
                model += (
                    vp.var_scl(v, "in")
                    == vp.var_scl((i0, v), "out") + params.Csw,
                    f"scl_mul_{v}_in_{i0}_plain_eq",
                )
                continue
            i1 = ilist[0]
        else:
            i0, i1 = ilist[0], ilist[1]
        model += (
            vp.var_scl(v, "in")
            == vp.var_scl((i0, v), "out") + vp.var_scl((i1, v), "out"),
            f"scl_mul_{v}_in_{i0}_{i1}_eq",
        )

    # add node rescale/bootstrap constraints
    for v in tdag.nodes:
        if tdag.nodes[v]["op"] == "constant":
            continue
        use_b = vp.var_use(v, "b")
        if tdag.nodes[v]["op"] == "input":
            model += (
                vp.var_lvl(v, "out") == vp.var_lvl(v, "in"),
                f"input_anchor_{v}_lvl",
            )
            model += (
                vp.var_scl(v, "out") == vp.var_scl(v, "in"),
                f"input_anchor_{v}_scl",
            )
            model += (vp.var_use(v, "b") == 0, f"input_anchor_{v}_bts")
            model += (vp.var_use(v, "r") == 0, f"input_anchor_{v}_rescale")
        model += (
            vp.var_scl(v, "in")
            <= params.Sf * (vp.var_lvl(v, "in") - params.lvl_lb + 2) - 7,
            f"decryptable_{v}_in",
        )
        model += (
            vp.var_scl(v, "out")
            <= params.Sf * (vp.var_lvl(v, "out") - params.lvl_lb + 2) - 7,
            f"decryptable_{v}_out",
        )
        # A compressed layer node may represent a ct-pt multiply without an
        # explicit constant predecessor. Keep every node boundary in the region
        # that can either rescale directly or reach the bootstrap lower bound.
        model += (
            vp.var_scl(v, "in")
            <= params.Sf * (vp.var_lvl(v, "in") - params.bts_lb + 1),
            f"bootstrap_feasible_{v}_in",
        )
        model += (
            vp.var_scl(v, "out")
            <= params.Sf * (vp.var_lvl(v, "out") - params.bts_lb + 1),
            f"bootstrap_feasible_{v}_out",
        )
        model += (
            vp.var_scl(v, "in") - params.Sf * vp.var_lvl(v, "in")
            <= params.Sf * (1 - params.bts_lb) + big_m_scale * (1 - use_b),
            f"bts_{v}_input_lvl_scl",
        )
        model += (
            vp.var_lvl(v, "out") >= params.bts_lb + 1 - big_m_level * (1 - use_b),
            f"bts_{v}_output_lvl",
        )
        model += (
            vp.var_scl(v, "out") >= params.Sf - big_m_scale * (1 - use_b),
            f"bts_{v}_output_scl",
        )
        # add not using bootstrapping constraints
        model += (
            vp.var_lvl(v, "out") - vp.var_lvl(v, "in") + vp.var_use(v, "r")
            <= big_m_level * use_b,
            f"nobts_{v}_lvl",
        )
        model += (
            vp.var_scl(v, "out")
            - vp.var_scl(v, "in")
            + params.Sf * vp.var_use(v, "r")
            >= -big_m_scale * use_b,
            f"nobts_{v}_scl",
        )

    # add edge rescale constraints
    for u, v in tdag.edges:
        if tdag.nodes[u]["op"] == "constant":
            continue
        edge_label = vp.get_edge_label(u, v)
        model += (
            vp.var_lvl((u, v), "out")
            <= vp.var_lvl((u, v), "in") - vp.var_use((u, v), "r"),
            f"edge_rescale_{edge_label}_lvl",
        )
        model += (
            vp.var_scl((u, v), "out")
            >= vp.var_scl((u, v), "in") - params.Sf * vp.var_use((u, v), "r"),
            f"edge_rescale_{edge_label}_scl",
        )


def add_ilp_linear_cost(tdag: Tdag, vp: VarPool, le: LatencyEstimator):
    cost_rescale_s = le.lin_op_lmaps["rescale_single"]
    cost_bts_s = le.lin_op_lmaps["bootstrap_single"]
    vp.rescale_cost = cost_rescale_s[1]

    # add node rescale/bootstrap costs
    for v in tdag.nodes:
        if tdag.nodes[v]["op"] == "constant":
            continue
        vp.total_cost.append(tdag.nodes[v]["weight"] * cost_bts_s[1] * vp.var_use(v, "b"))
        vp.total_cost.append(tdag.nodes[v]["weight"] * cost_rescale_s[1] * vp.var_use(v, "r"))
    # add edge rescale costs
    for u, v in tdag.edges:
        if tdag.nodes[u]["op"] == "constant":
            continue
        vp.total_cost.append(tdag.edges[u, v]["weight"] * cost_rescale_s[1] * vp.var_use((u, v), "r"))
    for v in tdag.nodes:
        single_cnt, double_cnt = tdag.get_v_weights(v)
        if single_cnt > 0:
            this_cost = le.lin_op_lmaps[tdag.nodes[v]["op"] + "_single"]
            vp.total_cost.append(single_cnt * (this_cost[0] * vp.var_lvl(v, "in") + this_cost[1]))
        if double_cnt > 0:
            this_cost = le.lin_op_lmaps[tdag.nodes[v]["op"] + "_double"]
            vp.total_cost.append(double_cnt * (this_cost[0] * vp.var_lvl(v, "in") + this_cost[1]))


def add_ilp_io_budgets(tdag: Tdag, vp: VarPool, io_budgets):
    v_in = list(tdag.inputs)[0]
    v_out = list(tdag.outputs)[0]
    model = vp.model
    params = vp.params
    if "in_lvl" in io_budgets and io_budgets["in_lvl"] >= 0:
        in_lvl = io_budgets["in_lvl"]
        model += vp.var_lvl(v_in, "in") == in_lvl, f"input_level_{in_lvl}"
    if "in_scl" in io_budgets and io_budgets["in_scl"] >= 0:
        in_scl = io_budgets["in_scl"]
        model += vp.var_scl(v_in, "in") == in_scl, f"input_scale_{in_scl}"
    if "out_lvl" in io_budgets and io_budgets["out_lvl"] >= 0:
        out_lvl = io_budgets["out_lvl"]
        model += vp.var_lvl(v_out, "out") == out_lvl, f"output_level_{out_lvl}"

    model += (
        vp.var_scl(v_out, "out") <= params.Sf * (vp.var_lvl(v_out, "out") + 1) - 7,
        "lattigo_output_scale_level_relation",
    )

    vp.total_cost.append(
        0.2 * vp.rescale_cost * tdag.get_full_size() * vp.var_scl(v_out, "out")
    )


def _solve_with_cbc(model: pulp.LpProblem, num_threads: int) -> int:
    solver = pulp.PULP_CBC_CMD(msg=False, threads=num_threads, gapRel=0.01)
    status = model.solve(solver)
    model._orbit_status = status
    return status


def solve_ilp_core(vp: VarPool, num_threads: int):
    model = vp.model
    model += pulp.lpSum(vp.total_cost)
    _solve_with_cbc(model, num_threads)


def solve_ilp_core_bypass(tdag: Tdag, vp: VarPool, io_budgets, num_threads: int):
    model = vp.model
    params = vp.params
    model += pulp.lpSum(vp.total_cost)

    v_main_o = io_budgets["maino_v"]
    assert v_main_o in tdag.nodes, f"Main output node {v_main_o} not in Tdag {tdag.name}"
    main_dag_size = io_budgets["main_dag_size"]
    main_qbp_cost = io_budgets["main_qbp_cost"]
    all_out = list(tdag.outputs)[0]

    # try all possible main-out (lvl, scl) configurations
    min_cost = None
    min_cost_ls = None

    def add_cons(mo_lvl, mo_scl):
        lvl_name = f"main_output_level_{mo_lvl}_{mo_scl}"
        scl_name = f"main_output_scale_{mo_lvl}_{mo_scl}"
        model.addConstraint(vp.var_lvl(v_main_o, "in") == mo_lvl, name=lvl_name)
        model.addConstraint(vp.var_scl(v_main_o, "in") == mo_scl, name=scl_name)
        return lvl_name, scl_name

    for (mo_lvl, mo_scl), main_cost in main_qbp_cost.items():
        tmp_cons_list = add_cons(mo_lvl, mo_scl)
        status = _solve_with_cbc(model, num_threads)
        if status == pulp.LpStatusOptimal:
            ft_cost = (
                0.2
                * vp.rescale_cost
                * main_dag_size
                * round(pulp.value(vp.var_scl(all_out, "out")))
            )
            this_cost = pulp.value(model.objective) + main_cost + ft_cost
            if (min_cost is None) or (this_cost < min_cost):
                min_cost = this_cost
                min_cost_ls = (mo_lvl, mo_scl)
        for constr_name in tmp_cons_list:
            del model.constraints[constr_name]

    if min_cost_ls is None:
        model._orbit_status = pulp.LpStatusInfeasible
        return
    add_cons(min_cost_ls[0], min_cost_ls[1])
    _solve_with_cbc(model, num_threads)


def solve_ilp(
    tdag: Tdag,
    io_budgets: dict,
    le: LatencyEstimator,
    task_name: str,
    num_threads: int,
    params: Params,
) -> tuple[Assign | None, float | None]:
    model = pulp.LpProblem(task_name.replace(" ", "_"), pulp.LpMinimize)
    vp = VarPool(tdag, params, model)

    add_ilp_constraints(tdag, vp)
    add_ilp_linear_cost(tdag, vp, le)
    add_ilp_io_budgets(tdag, vp, io_budgets)
    if "maino_v" in io_budgets:
        solve_ilp_core_bypass(tdag, vp, io_budgets, num_threads)
    else:
        solve_ilp_core(vp, num_threads)

    if getattr(model, "_orbit_status", None) != pulp.LpStatusOptimal:
        return None, None
    assign = decode_ilp_sol(tdag, vp)
    if "maino_v" in io_budgets:
        v_main_o = io_budgets["maino_v"]
        mo_lvl = assign.v_lvl_in[v_main_o]
        mo_scl = assign.v_scl_in[v_main_o]
        assert (mo_lvl, mo_scl) in io_budgets["main_qbp_cost"], (
            f"Main output (lvl, scl)=({mo_lvl}, {mo_scl}) not in provided QBP costs"
        )

    assign_cost = estimate_assign(assign, le)
    return assign, assign_cost


def decode_ilp_sol(tdag: Tdag, vp: VarPool) -> Assign:
    assign = Assign(tdag)
    params = vp.params

    def sol_value(var):
        value = pulp.value(var)
        if value is None:
            raise ValueError(f"Variable/expression {var} has no solution value")
        return round(value)

    for v in tdag.nodes:
        if tdag.nodes[v]["op"] == "input":
            # input nodes, need to store in-level/scale
            assign.v_lvl_in[v] = sol_value(vp.var_lvl(v, "in"))
            assign.v_scl_in[v] = sol_value(vp.var_scl(v, "in"))
            assert assign.v_scl_in[v] >= params.Sw, (
                f"Node {v} input scale {assign.v_scl_in[v]} below Sw={params.Sw}"
            )
        assign.v_lvl_out[v] = sol_value(vp.var_lvl(v, "out"))
        assign.v_scl_out[v] = sol_value(vp.var_scl(v, "out"))
        if tdag.nodes[v]["op"] != "constant":
            assert assign.v_scl_out[v] >= params.Sw, (
                f"Node {v} output scale {assign.v_scl_out[v]} below Sw={params.Sw}"
            )

    for u, v in tdag.edges:
        if tdag.nodes[u]["op"] == "constant":
            continue
        assign.e_lvl_out[(u, v)] = sol_value(vp.var_lvl((u, v), "out"))
        assign.e_scl_out[(u, v)] = sol_value(vp.var_scl((u, v), "out"))
        assert assign.e_scl_out[(u, v)] >= params.Sw, (
            f"Edge ({u},{v}) output scale {assign.e_scl_out[(u, v)]} below Sw={params.Sw}"
        )

    return assign
