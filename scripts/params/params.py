import json
import numpy as np
from ..resilience import ResilienceProfile

class Params:
    def __init__(
        self,
        le_json,
        sysname,
        mode,
        Sw=None,
        CSw=None,
        bpsdepth=None,
        threads=None,
        comp=None,
        part=None,
        reqbp=None,
        netname=None,
        resilience_profile=None,
    ):
        if le_json is None:
            return # should be filled later
        json_parsed = {}
        try:
            with open(le_json, 'r') as file:
                json_parsed = json.load(file)
        except FileNotFoundError:
            raise Exception(f"The file {le_json} was not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON from the file {le_json}.")
        
        self.poly_deg = int(json_parsed.get("poly_deg", 32768))
        self.max_slot = self.poly_deg // 2
        self.bts_ub = int(json_parsed.get("bootstrapLevelUpperBound", 14))
        self.bts_lb = int(json_parsed.get("bootstrapLevelLowerBound", 1))
        self.lvl_ub = int(json_parsed.get("levelUpperBound", 14))
        self.lvl_lb = int(json_parsed.get("levelLowerBound", 1))
        self.Sf = int(json_parsed.get("rescalingFactor", 51))
        self.backend = json_parsed.get("runtime", "")
        self.latency_table = json_parsed.get("latencyTable", {})
        
        assert mode in ["compile", "execute"], "Mode must be either 'compile' or 'execute'"
        self.mode = mode
        self.sysname = sysname
        self.Sw = Sw if Sw is not None else self.Sf
        self.Csw = CSw if CSw is not None else self.Sw
        self.bpsdepth = bpsdepth  # possibly None
        self.threads = threads if threads is not None else 16
        self.comp = comp if comp is not None else True
        self.part = part if part is not None else True
        self.reqbp = reqbp if reqbp is not None else False
        self.netname = netname if netname is not None else ""
        self.resilience_profile_path = resilience_profile
        self.resilience_profile = ResilienceProfile.load(resilience_profile)
        
        self.trunc_val = 1 # truncation value for latency estimation
        self.dacapo_mlir_in = True  # need to revert the input MLIR level
        self.dacapo_mlir_out = True # need to revert the output MLIR level

    def has_resilience_constraints(self) -> bool:
        return (
            self.resilience_profile is not None
            and len(self.resilience_profile.constraints) > 0
        )

    def scale_lower_bound(self, node_label: str, node_attrs: dict, port: str) -> int:
        if self.resilience_profile is None:
            return self.Sw
        return self.resilience_profile.scale_lower_bound(
            node_label,
            node_attrs,
            self.Sw,
            port,
        )
        
    def check_res(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
        if in_lvl < out_lvl:
            return False
        if self.Sf * in_lvl - in_scl < self.Sf * out_lvl - out_scl:
            return False
        return True
    
    def check_resbts(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
        if self.check_res(in_lvl, in_scl, out_lvl, out_scl):
            return True
        if not self.check_res(in_lvl, in_scl, self.bts_lb, self.Sf):
            return False
        r = max(0, round(np.ceil((self.Sf - out_scl) / self.Sf)))
        if not (self.bts_lb < out_lvl + r <= self.bts_ub):
            return False
        if not self.check_res(out_lvl+r, self.Sf, out_lvl, out_scl):
            return False
        return True
