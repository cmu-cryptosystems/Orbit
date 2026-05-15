from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params
from .qbp import QBP
from .ilp_worker import ILP_Worker
from .openevolve_backend import OpenEvolvePlacementWorker

import os
import sys

class QBPManager:
    def __init__(self, params: Params, le: LatencyEstimator):
        self.params = params
        self.le = le
        self.qbps: dict[str, QBP] = dict() # dag name -> QBP
        self.pdag_name_to_qbp = dict()  # pdag name -> (bj_qbp, bj_label)
        self.ilp_worker = (
            OpenEvolvePlacementWorker(params, le)
            if params.placement_backend == "openevolve"
            else ILP_Worker(params, le)
        )
        self.openevolve_diagnostics = []
    
    def save_qbps(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        for qbp_name, qbp in self.qbps.items():
            if len(qbp.dag.nodes) >= 100:
                qbp.save_qbp(os.path.join(dirpath, qbp_name))
    
    def load_qbps(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            if not os.path.isfile(filepath):
                continue
            try:
                qbp = QBP.load_qbp(filepath, self.params)
                self.qbps[qbp.dag.name] = qbp
            except Exception as e:
                print(f"Warning: Failed to load QBP from {filepath}: {e}", file=sys.stderr)
    
    def _get_qbp_bj(self, pdag: Tdag) -> tuple[QBP, dict[str, str]]:
        if pdag.name in self.pdag_name_to_qbp:
            return self.pdag_name_to_qbp[pdag.name]
        if self.params.has_resilience_constraints():
            qbp = self.qbps.get(pdag.name)
            if qbp is None:
                qbp = QBP(pdag)
                self.qbps[pdag.name] = qbp
            bj_label = {v: v for v in pdag.nodes}
            self.pdag_name_to_qbp[pdag.name] = (qbp, bj_label)
            return qbp, bj_label
        bj_label = None
        bj_qbp = None
        for dag_id, qbp in self.qbps.items():
            bj = tdag_equal_biject(pdag, qbp.dag)
            if bj is None:
                continue
            print(f"PDAG #{pdag.name} is equivalent to PDAG #{dag_id}")
            bj_qbp = qbp
            bj_label = bj
            break
        if bj_qbp is None:
            bj_qbp = QBP(pdag)
            bj_label = {v: v for v in pdag.nodes}
            self.qbps[pdag.name] = bj_qbp
        
        self.pdag_name_to_qbp[pdag.name] = (bj_qbp, bj_label)
        return bj_qbp, bj_label
    
    def _assign_biject(self, io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]], bj_qbp: QBP, bj_label: dict[str, str]) -> dict[tuple[int, int], dict[tuple[int, int], Assign]]:
        new_io_to_assign = dict()
        for in_key, out_to_assign in io_to_assign.items():
            new_out_to_assign = dict()
            for out_key, assign in out_to_assign.items():
                new_assign = Assign(bj_qbp.dag)
                for v in assign.v_lvl_out:
                    new_v = bj_label[v]
                    new_assign.v_lvl_out[new_v] = assign.v_lvl_out[v]
                for v in assign.v_scl_out:
                    new_v = bj_label[v]
                    new_assign.v_scl_out[new_v] = assign.v_scl_out[v]
                for v in assign.v_lvl_in:
                    new_v = bj_label[v]
                    new_assign.v_lvl_in[new_v] = assign.v_lvl_in[v]
                for v in assign.v_scl_in:
                    new_v = bj_label[v]
                    new_assign.v_scl_in[new_v] = assign.v_scl_in[v]
                for e in assign.e_lvl_out:
                    new_e = (bj_label[e[0]], bj_label[e[1]])
                    new_assign.e_lvl_out[new_e] = assign.e_lvl_out[e]
                for e in assign.e_scl_out:
                    new_e = (bj_label[e[0]], bj_label[e[1]])
                    new_assign.e_scl_out[new_e] = assign.e_scl_out[e]
                for v in assign.v_err_in:
                    new_v = bj_label[v]
                    new_assign.v_err_in[new_v] = assign.v_err_in[v]
                for v in assign.v_err_out:
                    new_v = bj_label[v]
                    new_assign.v_err_out[new_v] = assign.v_err_out[v]
                for e in assign.e_err_out:
                    new_e = (bj_label[e[0]], bj_label[e[1]])
                    new_assign.e_err_out[new_e] = assign.e_err_out[e]
                new_out_to_assign[out_key] = new_assign
            new_io_to_assign[in_key] = new_out_to_assign
        return new_io_to_assign
    
    def _merge_bypass_results(self, pdag: Tdag, fork_v: str, maino_v: str, 
                              main_io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]],
                              bypass_io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]]
                              ) -> tuple[dict[tuple[int, int], dict[tuple[int, int], Assign]], dict[tuple[int, int], dict[tuple[int, int], float]]]:
        merged_io_to_assign = dict()
        merged_io_to_cost = dict()
        for in_key, bps_out_to_assign in bypass_io_to_assign.items():
            new_assign = dict()
            new_cost = dict()
            for out_key, bps_assign in bps_out_to_assign.items():
                main_o_lvl = bps_assign.v_lvl_in.get(maino_v, None)
                main_o_scl = bps_assign.v_scl_in.get(maino_v, None)
                assert main_o_lvl is not None and main_o_scl is not None, f"Bypass assignment missing main output level/scale for node {maino_v}"
                
                this_main_assign = main_io_to_assign[in_key].get((main_o_lvl, main_o_scl), None)
                assert this_main_assign is not None, f"No main assignment found for input key {in_key} and output key {(main_o_lvl, main_o_scl)}"
                
                bps_assign_dict = bps_assign.to_dict()
                this_new_assign_dict = dict()
                for k, v in bps_assign_dict.items():
                    new_part_assign = getattr(this_main_assign, k).copy()
                    new_part_assign.update(v)
                    this_new_assign_dict[k] = new_part_assign
                this_new_assign = Assign.from_dict(pdag, this_new_assign_dict)
                
                this_new_assign.v_lvl_in = {fork_v: in_key[0]}
                this_new_assign.v_scl_in = {fork_v: in_key[1]}
                this_new_assign.v_lvl_out[maino_v] = bps_assign.v_lvl_out[maino_v]
                this_new_assign.v_scl_out[maino_v] = bps_assign.v_scl_out[maino_v]
                
                # Case 1: use main pass fork_v lvl/scl out as fork_v lvl/scl out
                this_new_assign.v_lvl_out[fork_v] = this_main_assign.v_lvl_out[fork_v]
                this_new_assign.v_scl_out[fork_v] = this_main_assign.v_scl_out[fork_v]
                try:
                    cost_use_main = estimate_assign(this_new_assign, self.le)
                except Exception as e:
                    print(f"Error estimating main cost: {e}", file=sys.stderr)
                    cost_use_main = float('inf')
                # Case 2: use bypass pass fork_v lvl/scl out as fork_v lvl/scl out
                this_new_assign.v_lvl_out[fork_v] = bps_assign.v_lvl_out[fork_v]
                this_new_assign.v_scl_out[fork_v] = bps_assign.v_scl_out[fork_v]
                try:
                    cost_use_bypass = estimate_assign(this_new_assign, self.le)
                except Exception as e:
                    print(f"Error estimating bypass cost: {e}", file=sys.stderr)
                    cost_use_bypass = float('inf')

                if cost_use_main <= cost_use_bypass:
                    this_new_assign.v_lvl_out[fork_v] = this_main_assign.v_lvl_out[fork_v]
                    this_new_assign.v_scl_out[fork_v] = this_main_assign.v_scl_out[fork_v]
                    cost = cost_use_main
                else:
                    cost = cost_use_bypass
                assert cost < float('inf'), f"Both main and bypass costs are infinite for input {in_key} and output {out_key}"
                new_assign[out_key] = this_new_assign
                new_cost[out_key] = cost
            merged_io_to_assign[in_key] = new_assign
            merged_io_to_cost[in_key] = new_cost
        return merged_io_to_assign, merged_io_to_cost
    
    def add_qbp(self, pdag: Tdag, in_budgets: dict[int, dict[int, float]]):
        print(f"Adding QBP for PDAG #{pdag.name}")
        print(f"  this_in_budgets: {sorted(in_budgets.items())}")
        bj_qbp, bj_label = self._get_qbp_bj(pdag)
        io_budgets_list = []
        num_total_tasks = len(in_budgets) * self.params.lvl_ub
        for in_lvl, in_scl_to_cost in in_budgets.items():
            for in_scl in in_scl_to_cost.keys():
                if (in_lvl, in_scl) not in bj_qbp.io_to_cost:
                    # need to solve placement for this input
                    if not self.params.part:
                        io_budgets_list.append({
                            'in_lvl': in_lvl,
                            'in_scl': in_scl
                        })
                    else:
                        for out_lvl in range(1, self.params.lvl_ub+1):
                            io_budgets_list.append({
                                'in_lvl': in_lvl,
                                'in_scl': in_scl,
                                'out_lvl': out_lvl
                            })
        num_remaining_tasks = len(io_budgets_list)
        print(f"  QBP Manager: Reusing Ratio: {1-num_remaining_tasks/num_total_tasks:.3f}.\n    Need to solve {num_remaining_tasks} / {num_total_tasks} placement tasks for PDAG #{pdag.name}")
        if len(io_budgets_list) == 0:
            return
        io_budgets_list = self._sample_openevolve_eval_budgets(io_budgets_list)
        io_to_assign, io_to_cost = self.ilp_worker.get_qbp(pdag, io_budgets_list)
        if getattr(self.params, "openevolve_evaluating_candidate", False):
            self.openevolve_diagnostics.append(getattr(self.ilp_worker, "last_diagnostics", {}))
        qbp_io_to_assign = self._assign_biject(io_to_assign, bj_qbp, bj_label)
        bj_qbp.append_io_results(io_to_cost, qbp_io_to_assign)
        
        this_qbp_cost = bj_qbp.get_all_costs()
        # print(f" ++++ Normal QBP ++++ ")
        # for (in_level, in_scale), out_costs in sorted(this_qbp_cost.items()):
        #     print(f"   + Input Level {in_level}, Input Scale {in_scale} output costs: {sorted(out_costs.items())}")

    
    def add_qbp_bypass(self, pdag: Tdag, main_pdag: Tdag, bypass_pdag: Tdag, in_budgets: dict[int, dict[int, float]]):
        dag_qbp, dag_label = self._get_qbp_bj(pdag)
        assert main_pdag.name in self.pdag_name_to_qbp, f"Main PDAG #{main_pdag.name} QBP not found in manager. Please add it first."
        main_qbp = self.pdag_name_to_qbp[main_pdag.name][0]
        # get main_pdag tolerance
        main_pdag_size = main_pdag.get_full_size()
        fork_v = list(pdag.inputs)[0]
        maino_v = list(main_pdag.outputs)[0]
        
        io_budgets_list = []
        for in_lvl, in_scl_to_cost in in_budgets.items():
            for in_scl in in_scl_to_cost.keys():
                if (in_lvl, in_scl) not in dag_qbp.io_to_cost:
                    main_o_costs = main_qbp.get_i_costs((in_lvl, in_scl))
                    if main_o_costs is None:
                        # not belong to current budget
                        continue
                    for dag_o_lvl in range(1, self.params.lvl_ub + 1):
                        io_budgets_list.append({
                            'in_lvl': in_lvl,
                            'in_scl': in_scl,
                            'out_lvl': dag_o_lvl,
                            'maino_v': maino_v,
                            'main_dag_size': main_pdag_size,
                            'main_qbp_cost': main_o_costs
                        })
        
        num_remain_tasks = len(io_budgets_list)
        print(f"  Bypass QBP Manager: Need to solve {num_remain_tasks} bypass placement tasks for PDAG #{pdag.name}")
        
        if len(io_budgets_list) == 0:
            return
        io_budgets_list = self._sample_openevolve_eval_budgets(io_budgets_list)
        
        bypass_io_to_assign, _ = self.ilp_worker.get_qbp(bypass_pdag, io_budgets_list)
        if getattr(self.params, "openevolve_evaluating_candidate", False):
            self.openevolve_diagnostics.append(getattr(self.ilp_worker, "last_diagnostics", {}))
        main_io_to_assign = self.get_qbp_assign(main_pdag)
        dag_io_to_assign, dag_io_to_cost = self._merge_bypass_results(pdag, fork_v, maino_v, main_io_to_assign, bypass_io_to_assign)
        dag_qbp_io_to_assign = self._assign_biject(dag_io_to_assign, dag_qbp, dag_label)
        dag_qbp.append_io_results(dag_io_to_cost, dag_qbp_io_to_assign)
    
    def add_qbp_existing(self, pdag: Tdag, io_to_cost: dict[tuple[int, int], dict[tuple[int, int], float]], io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]]):
        bj_qbp, bj_label = self._get_qbp_bj(pdag)
        qbp_io_to_assign = self._assign_biject(io_to_assign, bj_qbp, bj_label)
        bj_qbp.append_io_results(io_to_cost, qbp_io_to_assign)

    def _sample_openevolve_eval_budgets(self, io_budgets_list: list[dict]) -> list[dict]:
        if not getattr(self.params, "openevolve_evaluating_candidate", False):
            return io_budgets_list
        suite = getattr(self.params, "openevolve_eval_suite", "polybert-sampled")
        if suite == "polybert-full":
            return io_budgets_list
        if suite == "toy":
            return io_budgets_list[: min(len(io_budgets_list), 8)]
        if len(io_budgets_list) <= 64:
            return io_budgets_list
        max_sample = 48
        keep_levels = {
            1,
            max(1, self.params.bts_lb + 1),
            max(1, self.params.lvl_ub // 2),
            self.params.lvl_ub,
        }
        sampled = []
        seen = set()

        def add(budget: dict) -> None:
            if len(sampled) >= max_sample:
                return
            key = (
                int(budget.get("in_lvl", -1)),
                int(budget.get("in_scl", -1)),
                int(budget.get("out_lvl", -1)),
                str(budget.get("maino_v", "")),
            )
            if key in seen:
                return
            seen.add(key)
            sampled.append(budget)

        def budget_cost(budget: dict) -> float:
            costs = budget.get("main_qbp_cost")
            if isinstance(costs, dict) and costs:
                return float(min(costs.values()))
            return 0.0

        def budget_size(budget: dict) -> float:
            try:
                return float(budget.get("main_dag_size", 0.0))
            except (TypeError, ValueError):
                return 0.0

        def add_quantiles(budgets: list[dict], key_fn, limit: int = 5) -> None:
            if not budgets:
                return
            ordered = sorted(
                budgets,
                key=lambda item: (
                    key_fn(item),
                    int(item.get("out_lvl", -1)),
                    int(item.get("in_lvl", -1)),
                    int(item.get("in_scl", -1)),
                    str(item.get("maino_v", "")),
                ),
            )
            indexes = {0, len(ordered) // 4, len(ordered) // 2, (3 * len(ordered)) // 4, len(ordered) - 1}
            for idx in sorted(indexes)[:limit]:
                if 0 <= idx < len(ordered):
                    add(ordered[idx])

        by_level: dict[int, list[dict]] = {}
        bypass_budgets = []
        for budget in io_budgets_list:
            out_lvl = int(budget.get("out_lvl", -1))
            if out_lvl < 0:
                add(budget)
            if "maino_v" in budget:
                bypass_budgets.append(budget)
            by_level.setdefault(out_lvl, []).append(budget)

        add_quantiles(bypass_budgets, budget_cost, limit=8)
        add_quantiles(bypass_budgets, budget_size, limit=8)
        for out_lvl, budgets in sorted(by_level.items()):
            if out_lvl < 0:
                continue
            if out_lvl in keep_levels:
                add_quantiles(budgets, budget_cost, limit=5)
                add_quantiles(budgets, budget_size, limit=3)
            else:
                add_quantiles(budgets, budget_cost, limit=3)
            if len(sampled) >= max_sample:
                break

        if len(sampled) < max_sample:
            add_quantiles(io_budgets_list, budget_cost, limit=8)
            add_quantiles(io_budgets_list, budget_size, limit=8)
        if len(sampled) < max_sample:
            stride = max(1, len(io_budgets_list) // max_sample)
            for budget in io_budgets_list[::stride]:
                add(budget)
        return sampled or io_budgets_list[:1]
    
    def get_qbp_cost(self, pdag_name: str) -> dict:
        assert pdag_name in self.pdag_name_to_qbp, f"QBP for PDAG #{pdag_name} not found in manager."
        bj_qbp = self.pdag_name_to_qbp[pdag_name][0]
        return bj_qbp.get_all_costs()
    
    def get_qbp_assign(self, pdag: Tdag) -> dict:
        assert pdag.name in self.pdag_name_to_qbp, f"QBP for PDAG #{pdag.name} not found in manager."
        bj_qbp, bj_label = self.pdag_name_to_qbp[pdag.name]
        rev_bj_label = {v: k for k, v in bj_label.items()}
        io_to_assign = bj_qbp.get_all_assigns()
        rev_qbp = QBP(pdag)
        rev_io_to_assign = self._assign_biject(io_to_assign, rev_qbp, rev_bj_label)
        return rev_io_to_assign
