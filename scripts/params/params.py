import json
import numpy as np
from ..resilience import ResilienceProfile

DEFAULT_RESILIENCE_ERROR_MODEL = {
    "input_error_abs": 0.0,
    "add_error_abs": 1e-10,
    "mul_plain_error_abs": 1e-8,
    "mul_cipher_error_abs": 2e-8,
    "rotate_error_abs": 1e-9,
    "rescale_error_abs": 1e-8,
    "modswitch_error_abs": 1e-9,
    "upscale_error_abs": 0.0,
    "bootstrap_error_abs": 1e-6,
    "max_error_abs": 1.0,
}

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
        resilience_mode="waterline",
        allow_empty_resilience_match=False,
        resilience_decomposition="off",
        resilience_decompose_threshold=32,
        resilience_max_boundary_states=8,
        resilience_error_buckets=8,
        ilp_task_time_limit_sec=0.0,
        resilience_constraint_policy="relax-only",
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
        
        self.poly_deg = int(json_parsed.get("poly_deg", json_parsed.get("polynomialDegree", 32768)))
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
        self.requested_Sw = self.Sw
        self.requested_Csw = CSw
        self.bpsdepth = bpsdepth  # possibly None
        self.threads = threads if threads is not None else 16
        self.comp = comp if comp is not None else True
        self.part = part if part is not None else True
        self.reqbp = reqbp if reqbp is not None else False
        self.netname = netname if netname is not None else ""
        self.resilience_profile_path = resilience_profile
        self.resilience_profile = ResilienceProfile.load(resilience_profile)
        assert resilience_constraint_policy in ["relax-only", "hard-tau"], (
            "resilience_constraint_policy must be either 'relax-only' or 'hard-tau'"
        )
        self.resilience_constraint_policy = resilience_constraint_policy
        if (
            self.resilience_constraint_policy == "relax-only"
            and self.resilience_profile is not None
        ):
            relaxed_sw = self.resilience_profile.relaxed_global_scale(self.Sw)
            if relaxed_sw < self.Sw:
                print(
                    "Resilience relax-only policy lowered guided waterline "
                    f"from Sw={self.Sw} to Sw={relaxed_sw}."
                )
                self.Sw = relaxed_sw
                if self.requested_Csw is None:
                    self.Csw = relaxed_sw
        assert resilience_mode in ["waterline", "error-state"], (
            "resilience_mode must be either 'waterline' or 'error-state'"
        )
        self.resilience_mode = resilience_mode
        self.allow_empty_resilience_match = allow_empty_resilience_match
        assert resilience_decomposition in ["off", "bounded-dp"], (
            "resilience_decomposition must be either 'off' or 'bounded-dp'"
        )
        self.resilience_decomposition = resilience_decomposition
        self.resilience_decompose_threshold = resilience_decompose_threshold
        self.resilience_max_boundary_states = resilience_max_boundary_states
        self.resilience_error_buckets = resilience_error_buckets
        self.ilp_task_time_limit_sec = ilp_task_time_limit_sec
        self.resilience_match_report = None
        self.resilience_decomposition_report = None
        self.resilience_error_model = DEFAULT_RESILIENCE_ERROR_MODEL.copy()
        self.resilience_error_model.update(json_parsed.get("resilienceErrorModel", {}))
        
        self.trunc_val = 1 # truncation value for latency estimation
        self.dacapo_mlir_in = True  # need to revert the input MLIR level
        self.dacapo_mlir_out = True # need to revert the output MLIR level

    def has_resilience_constraints(self) -> bool:
        return (
            self.resilience_profile is not None
            and len(self.resilience_profile.constraints) > 0
        )

    def use_bounded_resilience_decomposition(self) -> bool:
        return (
            self.resilience_mode == "error-state"
            and self.resilience_decomposition == "bounded-dp"
            and self.has_resilience_constraints()
            and self.part
        )

    def scale_lower_bound(self, node_label: str, node_attrs: dict, port: str) -> int:
        if self.resilience_profile is None:
            return self.Sw
        profile_bound = self.resilience_profile.scale_lower_bound(
            node_label,
            node_attrs,
            self.Sw,
            port,
        )
        if self.resilience_constraint_policy == "relax-only":
            return min(self.Sw, profile_bound)
        return profile_bound
        
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
