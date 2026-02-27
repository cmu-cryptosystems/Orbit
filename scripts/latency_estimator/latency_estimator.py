import json

from more_itertools import strip
from ..params.params import Params
from ..utils.rot_decompose import get_naf_weight
from ..utils.linear_regression import linear_regression, linear_regression_max
import numpy as np


class LatencyEstimator:
    
    def _check_op_exist(self, op: str, lvl_lb: int, lvl_ub: int):
        if op not in self.op_lmaps.keys():
            raise Exception(f"Operation {op} not found in latency table.")
        lmap = self.op_lmaps[op]
        for i in range(lvl_lb, lvl_ub + 1):
            if i not in lmap.keys():
                raise Exception(f"Operation {op} does not have latency data for target level {i}.")
        
        for key in list(lmap.keys()):
            if key < lvl_lb or key > lvl_ub:
                del self.op_lmaps[op][key]
        return True
    
    def __init__(self, params: Params):
        self.params = params
        self.op_lmaps = dict()
        
        # load latency table
        for k, v in params.latency_table.items():
            op = k.split(".")[1]
            lmap = dict()
            for i, latency in enumerate(v):
                lmap[i+1] = latency
                # json file: starting from index=1
            self.op_lmaps[op] = lmap
        
        # check necessary ops
        self._check_op_exist('add_single', params.lvl_lb, params.lvl_ub)
        self._check_op_exist('add_double', params.lvl_lb, params.lvl_ub)
        self._check_op_exist('mul_single', params.lvl_lb, params.lvl_ub)
        self._check_op_exist('mul_double', params.lvl_lb, params.lvl_ub)
        self._check_op_exist('rotate_single', params.lvl_lb, params.lvl_ub)
        self._check_op_exist('negate_single', params.lvl_lb, params.lvl_ub)
        
        self._check_op_exist('modswitch_single', params.lvl_lb, params.lvl_ub-1)
        self._check_op_exist('rescale_single', params.lvl_lb, params.lvl_ub-1)
        self._check_op_exist('upscale_single', params.lvl_lb, params.lvl_ub)
        self._check_op_exist('bootstrap_single', params.bts_lb+1, params.bts_ub)
    
        # specify bts latency model
        if params.backend == 'Lattigo':
            # not support variadic bts latency model
            # use fixed latency model
            bts_sum = sum(self.op_lmaps['bootstrap_single'][i] for i in range(params.bts_lb+1, params.bts_ub+1))
            # range: [bts_lb+1, bts_ub]
            bts_cost = bts_sum / (params.bts_ub - params.bts_lb)
            for i in range(params.bts_lb+1, params.bts_ub+1):
                self.op_lmaps['bootstrap_single'][i] = bts_cost
        
        # generate linear regression models for Orbit
        if params.sysname == 'Orbit':
            self.lin_op_lmaps = dict()
            for op in ['add_single', 'add_double', 'mul_single', 'mul_double', 'rotate_single', 'negate_single']:
               self.lin_op_lmaps[op] = linear_regression(self.op_lmaps[op], params.lvl_lb, params.lvl_ub, params.trunc_val)
            
            self.lin_op_lmaps['rescale_single'] = linear_regression_max(self.op_lmaps['rescale_single'], params.lvl_lb, params.lvl_ub-1, params.trunc_val)
            if params.backend == 'Lattigo':
                self.lin_op_lmaps['bootstrap_single'] = linear_regression_max(self.op_lmaps['bootstrap_single'], params.bts_lb+1, params.bts_ub, params.trunc_val)
            else:
                self.lin_op_lmaps['bootstrap_single'] = linear_regression(self.op_lmaps['bootstrap_single'], params.bts_lb+1, params.bts_ub, params.trunc_val)
            # modswitch, upscale : ignored in Orbit

    def res_cost(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> float:
        if in_lvl < out_lvl:
            raise Exception(f"Cannot rescale from level {in_lvl} to higher level {out_lvl}.")
        if self.params.Sf * in_lvl - in_scl < self.params.Sf * out_lvl - out_scl:
            raise Exception(f"Cannot rescale from (level {in_lvl}, scale {in_scl}) to (level {out_lvl}, scale {out_scl}).")
        # first do modswitch to out_lvl + r, then do r times rescale to out_lvl, then do upscale
        r = max(0, round(np.ceil((in_scl - out_scl) / self.params.Sf)))
        total_cost = 0
        
        if out_lvl + r < in_lvl:
            total_cost += self.op_lmaps['modswitch_single'][out_lvl + r]
            in_lvl = out_lvl + r
        
        for i in range(r-1, -1, -1):
            this_target = min(2 * self.params.Sw, out_scl + self.params.Sf * i)
            if in_scl < this_target + self.params.Sf:
                total_cost += self.op_lmaps['upscale_single'][in_lvl]
                in_scl = this_target + self.params.Sf
            total_cost += self.op_lmaps['rescale_single'][in_lvl - 1]
            in_lvl -= 1
            in_scl -= self.params.Sf
        
        if in_scl < out_scl:
            total_cost += self.op_lmaps['upscale_single'][in_lvl]
            in_scl = out_scl
        assert in_lvl == out_lvl and in_scl == out_scl, f"Final level/scale mismatch: ({in_lvl}, {in_scl}) vs ({out_lvl}, {out_scl})"
        
        return total_cost
    
    def resbts_cost(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> float:
        if in_lvl >= out_lvl and self.params.Sf * in_lvl - in_scl >= self.params.Sf * out_lvl - out_scl:
            return self.res_cost(in_lvl, in_scl, out_lvl, out_scl)
        else:
            cost_before_bts = self.res_cost(in_lvl, in_scl, self.params.bts_lb, self.params.Sf)
            r = max(0, round(np.ceil((self.params.Sf - out_scl)/self.params.Sf)))
            cost_bts = self.op_lmaps['bootstrap_single'][out_lvl + r]
            cost_after_bts = self.res_cost(out_lvl + r, self.params.Sf, out_lvl, out_scl)
            return cost_before_bts + cost_bts + cost_after_bts