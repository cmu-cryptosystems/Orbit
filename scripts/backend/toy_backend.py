"""Synthetic cost model ("Toy" runtime) for Orbit without profiled Lattigo data.

Use :func:`toy_params` when calling the optimizer or building a :class:`~scripts.params.params.Params`
object, or pass ``cost_models/toy_backend.json`` to ``--costjson`` on the CLI.

The JSON sets ``"runtime": "Toy"`` so bootstrap latency uses the generic (non-Lattigo) path in
:class:`~scripts.latency_estimator.latency_estimator.LatencyEstimator`.
"""
from __future__ import annotations

import os
from typing import Any

_ORBIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOY_BACKEND_JSON = os.path.join(_ORBIT_ROOT, "cost_models", "toy_backend.json")


def toy_params(
    mode: str = "compile",
    *,
    sysname: str = "Orbit",
    cost_json: str | None = None,
    **kwargs: Any,
):
    """Build :class:`~scripts.params.params.Params` using the bundled toy cost model.

    Parameters match :class:`~scripts.params.params.Params` after ``le_json`` and ``sysname``:
    ``Sw``, ``CSw``, ``bpsdepth``, ``threads``, ``comp``, ``part``, ``reqbp``, ``netname``.
    """
    from ..params.params import Params

    path = cost_json if cost_json is not None else TOY_BACKEND_JSON
    return Params(path, sysname, mode, **kwargs)
