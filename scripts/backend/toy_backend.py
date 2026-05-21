from __future__ import annotations

from ..params.params import Params


def toy_params() -> Params:
    params = Params(None, "Orbit", "compile")
    params.poly_deg = 128
    params.max_slot = 64
    params.bts_ub = 14
    params.bts_lb = 1
    params.lvl_ub = 14
    params.lvl_lb = 1
    params.Sf = 51
    params.Sw = 51
    params.Csw = 51
    params.backend = "toy"
    params.latency_table = {}
    params.mode = "compile"
    params.sysname = "Orbit"
    params.bpsdepth = None
    params.threads = 1
    params.comp = False
    params.part = False
    params.reqbp = False
    params.netname = "toy"
    params.trunc_val = 1
    params.dacapo_mlir_in = False
    params.dacapo_mlir_out = False
    return params
