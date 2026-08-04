from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

from queue import Queue, Empty
import threading
import time
import traceback
import numpy as np

from .ilp_core import solve_ilp

class ILP_Worker:
    def __init__(self, params: Params, le: LatencyEstimator):
        self.params = params
        self.le = le
    def _lsabts_worker(self, ilp_threads: int, pdag: Tdag, task_queue: Queue, lock: threading.Lock, io_to_assign: dict, io_to_cost: dict):
        pdag_vin = list(pdag.inputs)[0]
        pdag_vout = list(pdag.outputs)[0]
        while not task_queue.empty():
            # Separate the get() from processing: under high thread contention
            # another worker can drain the queue between the empty() check above
            # and this get(), raising queue.Empty. Handling it here (instead of
            # in the processing except/finally) keeps task_done() balanced with
            # get() and avoids referencing an unassigned task_name.
            try:
                io_budget = task_queue.get(timeout=3)
            except Empty:
                break
            try:
                task_name = f"Partition_{pdag.name}"
                io_budget_name = (
                    f"in_lvl={io_budget.get('in_lvl', -1)}"
                    f"_in_scl={io_budget.get('in_scl', -1)}"
                    f"_out_lvl={io_budget.get('out_lvl', -1)}"
                )
                if 'maino_v' in io_budget:
                    mq = io_budget["main_qbp_cost"]
                    key_part = "_".join(f"{a}_{b}" for a, b in sorted(mq.keys()))
                    io_budget_name += f"_main_qbp_{key_part}"
                task_name = f"Partition_{pdag.name}_{io_budget_name}"
                pasn, pasn_cost = solve_ilp(pdag, io_budget, self.le, task_name, ilp_threads, self.params)
                if pasn is not None:
                    real_in_lvl = pasn.v_lvl_in[pdag_vin]
                    real_in_scale = pasn.v_scl_in[pdag_vin]
                    real_out_lvl = pasn.v_lvl_out[pdag_vout]
                    real_out_scale = pasn.v_scl_out[pdag_vout]
                    if 'in_lvl' in io_budget and io_budget['in_lvl'] >= 0:
                        assert io_budget['in_lvl'] == real_in_lvl, f"Input level mismatch: {io_budget['in_lvl']} vs {real_in_lvl}"
                    if 'in_scl' in io_budget and io_budget['in_scl'] >= 0:
                        assert io_budget['in_scl'] == real_in_scale, f"Input scale mismatch: {io_budget['in_scl']} vs {real_in_scale}"
                    if 'out_lvl' in io_budget and io_budget['out_lvl'] >= 0:
                        assert io_budget['out_lvl'] == real_out_lvl, f"Output level mismatch: {io_budget['out_lvl']} vs {real_out_lvl}"
                    
                    in_key = (real_in_lvl, real_in_scale)
                    out_key = (real_out_lvl, real_out_scale)
                    
                    with lock:
                        if in_key not in io_to_cost:
                            io_to_cost[in_key] = {out_key: pasn_cost}
                            io_to_assign[in_key] = {out_key: pasn}
                        else:
                            io_to_cost[in_key][out_key] = pasn_cost
                            io_to_assign[in_key][out_key] = pasn
            except Exception as e:
                print(f"Error processing {task_name}: {e}")
                traceback.print_exc()
            finally:
                task_queue.task_done()
    
    def get_qbp(self, pdag: Tdag, io_budgets_list: list[dict]
                )-> tuple[dict[tuple[int, int], dict[tuple[int, int], Assign]], dict[tuple[int, int], dict[tuple[int, int], float]]]:
        task_queue = Queue()
        for io_budget in io_budgets_list:
            task_queue.put(io_budget)
        
        io_to_assign = dict()
        io_to_cost = dict()
        lock = threading.Lock()
        
        threads = []
        num_workers = min(self.params.threads, len(io_budgets_list))
        ilp_threads = round(np.ceil(self.params.threads / num_workers))
        
        for _ in range(num_workers):
            t = threading.Thread(target=self._lsabts_worker, args=(ilp_threads, pdag, task_queue, lock, io_to_assign, io_to_cost))
            t.start()
            threads.append(t)
        
        task_queue.join()
        for t in threads:
            t.join()
        
        return io_to_assign, io_to_cost
    
    
    

