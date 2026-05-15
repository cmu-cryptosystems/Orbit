from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


ESTIMATOR_SOURCE = "tuneinsight/ckks-noise-estimator"
ESTIMATOR_PAPER = "https://eprint.iacr.org/2024/853"
ESTIMATOR_MODEL = "tuneinsight-lattigo-v6-heuristic"
ESTIMATOR_VERSION = "orbit-estimator-gate-v1"


def estimate_compile_result_noise(
    context: dict[str, Any],
    result: dict[str, Any],
    params: Any,
) -> dict[str, Any]:
    """Estimate output precision for a replayed OpenEvolve compile candidate.

    This is a deterministic gate shaped after Tune Insight's CKKS estimator:
    it models canonical-embedding rounding, rescale, rotation/key-switching,
    multiplication/relinearization, and bootstrapping costs in log2-error space.
    A compiled sidecar can replace it through ``--noise-estimator-binary`` while
    preserving this JSON contract.
    """

    request = _estimator_request(context, result, params)
    binary = getattr(params, "noise_estimator_binary", None)
    if binary:
        sidecar = _run_sidecar(binary, request, getattr(params, "noise_estimator_timeout_sec", 30))
        if sidecar is not None:
            return _normalize_estimate(sidecar, params)
        request["sidecar_error"] = "sidecar_failed_or_returned_bad_json"
    estimate = _heuristic_estimate(request)
    return _normalize_estimate(estimate, params)


def _estimator_request(context: dict[str, Any], result: dict[str, Any], params: Any) -> dict[str, Any]:
    ckks = context.get("ckks", {})
    selected = result.get("selected_output_state") or {}
    out_scale = _as_int(selected.get("out_scl"), _as_int(ckks.get("Sw"), getattr(params, "Sw", 40)))
    return {
        "schema_version": "orbit-noise-estimator-request-v1",
        "source": ESTIMATOR_SOURCE,
        "model": ESTIMATOR_MODEL,
        "paper": ESTIMATOR_PAPER,
        "ckks": {
            "poly_degree": _as_int(ckks.get("poly_degree"), getattr(params, "poly_deg", 32768)),
            "max_slots": _as_int(ckks.get("max_slots"), getattr(params, "max_slot", 16384)),
            "Sf": _as_int(ckks.get("Sf"), getattr(params, "Sf", 40)),
            "Sw": _as_int(ckks.get("Sw"), getattr(params, "Sw", 40)),
            "lvl_lb": _as_int(ckks.get("lvl_lb"), getattr(params, "lvl_lb", 1)),
            "lvl_ub": _as_int(ckks.get("lvl_ub"), getattr(params, "lvl_ub", 1)),
        },
        "candidate": {
            "final_latency_usec": _as_float(result.get("final_latency_usec"), float("inf")),
            "bootstrap_count": _as_int(result.get("bootstrap_count"), 0),
            "rescale_count": _as_int(result.get("rescale_count"), 0),
            "fallback_selected_budgets": _as_int(result.get("fallback_selected_budgets"), 0),
            "profile_risk": _as_float(result.get("profile_risk"), 0.0),
            "output_scale_bits": out_scale,
            "selected_output_state": selected,
            "bootstrap_locations": result.get("bootstrap_locations") or {},
            "rescale_locations": result.get("rescale_locations") or {},
            "bottleneck_summary": result.get("bottleneck_summary") or [],
        },
        "resilience": context.get("resilience", {}),
        "sidecar_error": None,
    }


def _run_sidecar(binary: str, request: dict[str, Any], timeout_sec: int) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [binary],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _heuristic_estimate(request: dict[str, Any]) -> dict[str, Any]:
    ckks = request["ckks"]
    candidate = request["candidate"]
    poly_degree = max(2, _as_int(ckks.get("poly_degree"), 32768))
    scale_bits = max(1, _as_int(candidate.get("output_scale_bits"), _as_int(ckks.get("Sw"), 40)))
    rescale_count = max(0, _as_int(candidate.get("rescale_count"), 0))
    bootstrap_count = max(0, _as_int(candidate.get("bootstrap_count"), 0))
    fallback_count = max(0, _as_int(candidate.get("fallback_selected_budgets"), 0))
    profile_risk = max(0.0, _as_float(candidate.get("profile_risk"), 0.0))

    logn = math.log2(poly_degree)
    canonical_sigma_bits = 0.5 * max(0.0, logn - 1.0)
    rounding_cost = math.sqrt(max(1.0, rescale_count)) * (2.0 ** canonical_sigma_bits)
    keyswitch_cost = _location_weight(candidate.get("rescale_locations")) * 0.125
    bootstrap_cost = bootstrap_count * (2.0 ** max(0.0, canonical_sigma_bits - 2.0))
    profile_cost = math.log2(1.0 + profile_risk) if profile_risk > 0.0 else 0.0
    aggregate_cost = 1.0 + rounding_cost + keyswitch_cost + bootstrap_cost + profile_cost

    estimated_noise_bits = math.log2(max(aggregate_cost, 1.0))
    estimated_precision_bits = max(0.0, scale_bits - estimated_noise_bits)
    output_margin_bits = estimated_precision_bits - 2.0
    hotspots = _hotspots(candidate)
    unsupported = []
    if fallback_count > 0:
        unsupported.append("seed_fallback_selected")
    if request.get("sidecar_error"):
        unsupported.append(str(request["sidecar_error"]))

    return {
        "schema_version": "orbit-noise-estimator-result-v1",
        "source": ESTIMATOR_SOURCE,
        "paper": ESTIMATOR_PAPER,
        "model": ESTIMATOR_MODEL,
        "version": ESTIMATOR_VERSION,
        "mode": "internal_heuristic",
        "valid": True,
        "fallback": False,
        "estimated_noise_bits": estimated_noise_bits,
        "estimated_precision_bits": estimated_precision_bits,
        "output_margin_bits": output_margin_bits,
        "min_output_margin_bits": 2.0,
        "unsupported_ops": unsupported,
        "hotspots": hotspots,
        "ckks": ckks,
    }


def _normalize_estimate(data: dict[str, Any], params: Any) -> dict[str, Any]:
    min_margin = float(getattr(params, "noise_estimator_min_output_margin_bits", 2.0))
    margin = _as_float(data.get("output_margin_bits"), float("-inf"))
    unsupported = data.get("unsupported_ops") or []
    if not isinstance(unsupported, list):
        unsupported = [str(unsupported)]
    fallback = bool(data.get("fallback", False))
    valid = bool(data.get("valid", True)) and margin >= min_margin and not fallback and not unsupported
    result = {
        **data,
        "schema_version": data.get("schema_version", "orbit-noise-estimator-result-v1"),
        "source": data.get("source", ESTIMATOR_SOURCE),
        "paper": data.get("paper", ESTIMATOR_PAPER),
        "model": data.get("model", ESTIMATOR_MODEL),
        "version": data.get("version", ESTIMATOR_VERSION),
        "min_output_margin_bits": min_margin,
        "output_margin_bits": margin,
        "estimated_noise_bits": _as_float(data.get("estimated_noise_bits"), 0.0),
        "estimated_precision_bits": _as_float(data.get("estimated_precision_bits"), 0.0),
        "unsupported_ops": unsupported,
        "fallback": fallback,
        "valid": valid,
    }
    if "hotspots" not in result:
        result["hotspots"] = []
    return result


def write_noise_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hotspots(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for kind, locations in (
        ("bootstrap", candidate.get("bootstrap_locations") or {}),
        ("rescale", candidate.get("rescale_locations") or {}),
    ):
        if not isinstance(locations, dict):
            continue
        for location, count in locations.items():
            entry = combined.setdefault(str(location), {"location": str(location)})
            entry[kind] = int(count)
    hotspots = list(combined.values())
    hotspots.sort(key=lambda item: (item.get("bootstrap", 0), item.get("rescale", 0)), reverse=True)
    return hotspots[:12]


def _location_weight(locations: Any) -> float:
    if not isinstance(locations, dict):
        return 0.0
    total = 0.0
    for count in locations.values():
        try:
            total += max(0.0, float(count))
        except (TypeError, ValueError):
            continue
    return total


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)
