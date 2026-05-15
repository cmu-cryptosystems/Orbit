from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ESTIMATOR_SOURCE = "tuneinsight/ckks-noise-estimator"
ESTIMATOR_PAPER = "https://eprint.iacr.org/2024/853"
ESTIMATOR_MODEL = "componentwise-average-case-ckks"
ESTIMATOR_VERSION = "orbit-estimator-gate-v2"

ROUNDING_VARIANCE = 1.0 / 12.0
PUBLIC_KEY_FRESH_VARIANCE_0 = 1.0 / 6.0
PUBLIC_KEY_FRESH_VARIANCE_1 = 1.0 / 12.0
DEFAULT_INPUT_STD = 1.0 / math.sqrt(3.0)
DEFAULT_ALPHA = 14.0
DEFAULT_FRESH_ENCRYPTION_SIGMA = 3.2
DEFAULT_PRECISION_RESERVE_BITS = 2.0


def estimate_compile_result_noise(
    context: dict[str, Any],
    result: dict[str, Any],
    params: Any,
) -> dict[str, Any]:
    """Estimate output precision for a replayed OpenEvolve compile candidate.

    This is a deterministic gate shaped after Tune Insight's CKKS estimator
    and the component-wise average-case CKKS noise model in ePrint 2024/853.
    When a replayed assignment is present it propagates coefficient variances
    through the assigned TDAG, then derives the final decoded infinity-norm
    bound only once with the paper's alpha-standard-deviation tail bound.
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
    estimate = _mathematical_estimate(request)
    return _normalize_estimate(estimate, params)


def _estimator_request(context: dict[str, Any], result: dict[str, Any], params: Any) -> dict[str, Any]:
    ckks = context.get("ckks") or context.get("constraints", {}).get("ckks", {})
    ckks = {
        "poly_degree": ckks.get("poly_degree", ckks.get("polynomial_degree", ckks.get("poly_deg"))),
        "max_slots": ckks.get("max_slots"),
        "Sf": ckks.get("Sf", ckks.get("rescaling_factor")),
        "Sw": ckks.get("Sw", ckks.get("input_waterline")),
        "lvl_lb": ckks.get("lvl_lb", ckks.get("level_lower_bound")),
        "lvl_ub": ckks.get("lvl_ub", ckks.get("level_upper_bound")),
        "bts_lb": ckks.get("bts_lb", ckks.get("bootstrap_level_lower_bound")),
        "bts_ub": ckks.get("bts_ub", ckks.get("bootstrap_level_upper_bound")),
        "fresh_encryption_sigma": ckks.get("fresh_encryption_sigma"),
        "secret_hamming_weight": ckks.get("secret_hamming_weight"),
    }
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
            "bts_lb": _as_int(ckks.get("bts_lb"), getattr(params, "bts_lb", 1)),
            "bts_ub": _as_int(ckks.get("bts_ub"), getattr(params, "bts_ub", 1)),
            "fresh_encryption_sigma": _as_float(
                ckks.get("fresh_encryption_sigma"),
                getattr(params, "noise_estimator_fresh_encryption_sigma", DEFAULT_FRESH_ENCRYPTION_SIGMA),
            ),
            "secret_hamming_weight": _as_int(
                ckks.get("secret_hamming_weight"),
                getattr(params, "noise_estimator_secret_hamming_weight", 0),
            ),
        },
        "tdag": context.get("tdag") or {},
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
            "assignment": result.get("assignment") or {},
        },
        "policy": {
            "alpha": _as_float(
                getattr(params, "noise_estimator_alpha", DEFAULT_ALPHA),
                DEFAULT_ALPHA,
            ),
            "input_std": _as_float(
                getattr(params, "noise_estimator_input_std", DEFAULT_INPUT_STD),
                DEFAULT_INPUT_STD,
            ),
            "precision_reserve_bits": _as_float(
                getattr(params, "noise_estimator_precision_reserve_bits", DEFAULT_PRECISION_RESERVE_BITS),
                DEFAULT_PRECISION_RESERVE_BITS,
            ),
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


@dataclass(frozen=True)
class _NoiseState:
    sigma0: float
    sigma1: float
    message_sigma: float
    scale_bits: int
    level: int
    encrypted: bool = True
    node: str | None = None


def _mathematical_estimate(request: dict[str, Any]) -> dict[str, Any]:
    unsupported = []
    if request.get("sidecar_error"):
        unsupported.append(str(request["sidecar_error"]))
    if _as_int(request["candidate"].get("fallback_selected_budgets"), 0) > 0:
        unsupported.append("seed_fallback_selected")

    assignment = request["candidate"].get("assignment")
    tdag = request.get("tdag") or {}
    if isinstance(assignment, dict) and assignment and isinstance(tdag, dict) and tdag.get("nodes"):
        estimate = _assignment_estimate(request, unsupported)
    else:
        estimate = _aggregate_bound_estimate(request, unsupported)
    return estimate


def _assignment_estimate(request: dict[str, Any], unsupported: list[str]) -> dict[str, Any]:
    ckks = request["ckks"]
    policy = request["policy"]
    candidate = request["candidate"]
    assignment = candidate.get("assignment") or {}
    tdag = request.get("tdag") or {}
    nodes = tdag.get("nodes") or {}
    edges = tdag.get("edges") or []
    topo = [str(node) for node in tdag.get("topological_order") or nodes.keys()]
    outputs = [str(node) for node in tdag.get("outputs") or []]
    if not outputs and topo:
        outputs = [topo[-1]]

    sf_bits = max(1, _as_int(ckks.get("Sf"), 40))
    sw_bits = max(1, _as_int(ckks.get("Sw"), sf_bits))
    input_std = max(0.0, _as_float(policy.get("input_std"), DEFAULT_INPUT_STD))
    fresh_sigma = max(
        0.0,
        _as_float(ckks.get("fresh_encryption_sigma"), DEFAULT_FRESH_ENCRYPTION_SIGMA),
    )

    incoming: dict[str, list[str]] = {str(node): [] for node in nodes}
    for edge in edges:
        u = str(edge.get("u"))
        v = str(edge.get("v"))
        if v in incoming:
            incoming[v].append(u)

    states: dict[str, _NoiseState] = {}
    trace: dict[str, dict[str, Any]] = {}
    try:
        for node in topo:
            attrs = nodes.get(node, {})
            op = str(attrs.get("op", ""))
            if op == "constant":
                states[node] = _constant_state(node, attrs, assignment, sw_bits)
                trace[node] = _trace_entry(op, states[node])
                continue

            if op == "input" or not incoming.get(node):
                raw = _fresh_input_state(node, assignment, sf_bits, sw_bits, input_std, fresh_sigma)
            else:
                pred_states = [
                    _state_on_edge(states[pred], pred, node, assignment, ckks)
                    for pred in incoming.get(node, [])
                    if pred in states
                ]
                raw = _apply_node_operation(node, attrs, pred_states, ckks, input_std)

            out_level = _map_int(assignment, "v_lvl_out", node, raw.level)
            out_scale = _map_int(assignment, "v_scl_out", node, raw.scale_bits)
            states[node] = _apply_transition(raw, out_level, out_scale, ckks, f"node:{node}")
            trace[node] = _trace_entry(op, states[node])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        fallback_unsupported = list(unsupported)
        estimate = _aggregate_bound_estimate(request, fallback_unsupported)
        estimate.setdefault("details", {})["assignment_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return estimate

    output_states = [states[node] for node in outputs if node in states]
    if not output_states:
        output_states = list(states.values())[-1:]
    bound = max(_decoded_bound_bits(state, ckks, policy) for state in output_states)

    hotspots = _hotspots(candidate)
    hotspots.extend(_noise_hotspots(trace, ckks, policy))
    return _estimate_payload(
        request,
        mode="internal_componentwise_average_case",
        unsupported=unsupported,
        estimated_noise_bits=bound["estimated_noise_bits"],
        estimated_precision_bits=bound["estimated_precision_bits"],
        hotspots=hotspots,
        details={
            "outputs": {
                state.node or f"output_{idx}": _decoded_bound_bits(state, ckks, policy)
                for idx, state in enumerate(output_states)
            },
            "assignment_replayed": True,
            "trace_node_count": len(trace),
        },
    )


def _aggregate_bound_estimate(request: dict[str, Any], unsupported: list[str]) -> dict[str, Any]:
    ckks = request["ckks"]
    policy = request["policy"]
    candidate = request["candidate"]
    scale_bits = max(1, _as_int(candidate.get("output_scale_bits"), _as_int(ckks.get("Sw"), 40)))
    sf_bits = max(1, _as_int(ckks.get("Sf"), 40))
    rescale_count = max(0, _as_int(candidate.get("rescale_count"), 0))
    bootstrap_count = max(0, _as_int(candidate.get("bootstrap_count"), 0))
    profile_risk = max(0.0, _as_float(candidate.get("profile_risk"), 0.0))

    fresh_sigma = max(
        0.0,
        _as_float(ckks.get("fresh_encryption_sigma"), DEFAULT_FRESH_ENCRYPTION_SIGMA),
    )
    input_std = max(0.0, _as_float(policy.get("input_std"), DEFAULT_INPUT_STD))
    initial_scale_bits = scale_bits + rescale_count * sf_bits
    state = _NoiseState(
        sigma0=fresh_sigma,
        sigma1=0.0,
        message_sigma=input_std * _pow2(initial_scale_bits),
        scale_bits=initial_scale_bits,
        level=max(1, _as_int(ckks.get("lvl_ub"), 1)),
        encrypted=True,
        node="aggregate",
    )
    for idx in range(rescale_count):
        state = _rescale_once(state, sf_bits, f"aggregate_rescale_{idx}")
    for idx in range(bootstrap_count):
        state = _bootstrap_state(state, ckks, f"aggregate_bootstrap_{idx}")

    if profile_risk > 0.0:
        risk_sigma = math.sqrt(math.log2(1.0 + profile_risk) + 1.0)
        state = _NoiseState(
            sigma0=_hypot(state.sigma0, risk_sigma),
            sigma1=state.sigma1,
            message_sigma=state.message_sigma,
            scale_bits=state.scale_bits,
            level=state.level,
            encrypted=state.encrypted,
            node=state.node,
        )

    bound = _decoded_bound_bits(state, ckks, policy)
    hotspots = _hotspots(candidate)
    return _estimate_payload(
        request,
        mode="internal_componentwise_aggregate_bound",
        unsupported=unsupported,
        estimated_noise_bits=bound["estimated_noise_bits"],
        estimated_precision_bits=bound["estimated_precision_bits"],
        hotspots=hotspots,
        details={"assignment_replayed": False, "aggregate_state": bound},
    )


def _estimate_payload(
    request: dict[str, Any],
    *,
    mode: str,
    unsupported: list[str],
    estimated_noise_bits: float,
    estimated_precision_bits: float,
    hotspots: list[dict[str, Any]],
    details: dict[str, Any],
) -> dict[str, Any]:
    policy = request["policy"]
    precision_reserve = max(
        0.0,
        _as_float(policy.get("precision_reserve_bits"), DEFAULT_PRECISION_RESERVE_BITS),
    )
    output_margin_bits = estimated_precision_bits - precision_reserve

    return {
        "schema_version": "orbit-noise-estimator-result-v1",
        "source": ESTIMATOR_SOURCE,
        "paper": ESTIMATOR_PAPER,
        "model": ESTIMATOR_MODEL,
        "version": ESTIMATOR_VERSION,
        "mode": mode,
        "valid": True,
        "fallback": False,
        "estimated_noise_bits": estimated_noise_bits,
        "estimated_precision_bits": estimated_precision_bits,
        "output_margin_bits": output_margin_bits,
        "precision_reserve_bits": precision_reserve,
        "min_output_margin_bits": DEFAULT_PRECISION_RESERVE_BITS,
        "unsupported_ops": unsupported,
        "hotspots": hotspots,
        "ckks": request["ckks"],
        "bound_model": _bound_model(request["ckks"], policy),
        "details": details,
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


def _fresh_input_state(
    node: str,
    assignment: dict[str, Any],
    sf_bits: int,
    sw_bits: int,
    input_std: float,
    fresh_sigma: float,
) -> _NoiseState:
    scale_bits = _map_int(assignment, "v_scl_in", node, sw_bits)
    level = _map_int(assignment, "v_lvl_in", node, _map_int(assignment, "v_lvl_out", node, 1))
    return _NoiseState(
        sigma0=fresh_sigma,
        sigma1=0.0,
        message_sigma=max(input_std, 0.0) * _pow2(scale_bits or sf_bits),
        scale_bits=scale_bits,
        level=level,
        encrypted=True,
        node=node,
    )


def _constant_state(
    node: str,
    attrs: dict[str, Any],
    assignment: dict[str, Any],
    sw_bits: int,
) -> _NoiseState:
    descr = attrs.get("op_descr") or {}
    scale_bits = _map_int(assignment, "v_scl_out", node, sw_bits)
    level = _map_int(assignment, "v_lvl_out", node, 1)
    rms_var = max(0.0, _as_float(descr.get("rms_var"), 0.0))
    value = abs(_as_float(descr.get("value"), 0.0))
    coefficient_std = math.sqrt(rms_var) if rms_var > 0.0 else value
    return _NoiseState(
        sigma0=math.sqrt(ROUNDING_VARIANCE),
        sigma1=0.0,
        message_sigma=coefficient_std * _pow2(scale_bits),
        scale_bits=scale_bits,
        level=level,
        encrypted=False,
        node=node,
    )


def _state_on_edge(
    state: _NoiseState,
    u: str,
    v: str,
    assignment: dict[str, Any],
    ckks: dict[str, Any],
) -> _NoiseState:
    key = _edge_key(u, v)
    out_level = _map_int(assignment, "e_lvl_out", key, state.level)
    out_scale = _map_int(assignment, "e_scl_out", key, state.scale_bits)
    return _apply_transition(state, out_level, out_scale, ckks, f"edge:{u}->{v}")


def _apply_node_operation(
    node: str,
    attrs: dict[str, Any],
    pred_states: list[_NoiseState],
    ckks: dict[str, Any],
    input_std: float,
) -> _NoiseState:
    op = str(attrs.get("op", ""))
    descr = attrs.get("op_descr") or {}
    weight = max(1, _as_int(attrs.get("weight"), 1))
    if not pred_states:
        return _NoiseState(
            sigma0=DEFAULT_FRESH_ENCRYPTION_SIGMA,
            sigma1=0.0,
            message_sigma=max(input_std, 0.0) * _pow2(_as_int(ckks.get("Sw"), 40)),
            scale_bits=_as_int(ckks.get("Sw"), 40),
            level=_as_int(ckks.get("lvl_ub"), 1),
            encrypted=True,
            node=node,
        )

    if op in ("negate", "output", "modswitch", "rescale"):
        state = pred_states[0]
    elif op == "upscale":
        state = _upscale_state(pred_states[0], max(0, _as_int(descr.get("upFactor"), 0)), node)
    elif op == "bootstrap":
        state = _bootstrap_state(pred_states[0], ckks, node)
    elif op == "rotate":
        state = pred_states[0]
        rotate_weight = max(1, _as_int(descr.get("weight"), weight))
        for _ in range(rotate_weight):
            state = _rotate_state(state, ckks, node)
    elif op == "add":
        state = _add_states(pred_states, node)
        for _ in range(max(0, _as_int(descr.get("double"), 0) - max(0, len([s for s in pred_states if s.encrypted]) - 1))):
            state = _add_states([state, state], node)
    elif op == "mul":
        state = _mul_states(pred_states, node, ckks)
        for _ in range(max(0, weight - 1)):
            state = _mul_states([state, pred_states[-1]], node, ckks)
    else:
        state = pred_states[0]
    return _NoiseState(
        sigma0=state.sigma0,
        sigma1=state.sigma1,
        message_sigma=state.message_sigma,
        scale_bits=state.scale_bits,
        level=state.level,
        encrypted=state.encrypted,
        node=node,
    )


def _add_states(states: list[_NoiseState], node: str) -> _NoiseState:
    encrypted = [state for state in states if state.encrypted]
    plaintext = [state for state in states if not state.encrypted]
    if not encrypted:
        sigma0 = math.sqrt(sum(state.sigma0 * state.sigma0 for state in states))
        return _NoiseState(
            sigma0=sigma0,
            sigma1=0.0,
            message_sigma=math.sqrt(sum(state.message_sigma * state.message_sigma for state in states)),
            scale_bits=max((state.scale_bits for state in states), default=0),
            level=min((state.level for state in states), default=0),
            encrypted=False,
            node=node,
        )

    scale_bits = max(state.scale_bits for state in states)
    level = min(state.level for state in states)
    sigma0_sq = 0.0
    sigma1_sq = 0.0
    message_sq = 0.0
    for state in encrypted:
        factor = _pow2(scale_bits - state.scale_bits)
        sigma0_sq += (state.sigma0 * factor) ** 2
        sigma1_sq += (state.sigma1 * factor) ** 2
        message_sq += (state.message_sigma * factor) ** 2
    for state in plaintext:
        factor = _pow2(scale_bits - state.scale_bits)
        sigma0_sq += (state.sigma0 * factor) ** 2
        message_sq += (state.message_sigma * factor) ** 2
    return _NoiseState(
        sigma0=math.sqrt(sigma0_sq),
        sigma1=math.sqrt(sigma1_sq),
        message_sigma=math.sqrt(message_sq),
        scale_bits=scale_bits,
        level=level,
        encrypted=True,
        node=node,
    )


def _mul_states(states: list[_NoiseState], node: str, ckks: dict[str, Any]) -> _NoiseState:
    if len(states) == 1:
        left = states[0]
        right = states[0]
    else:
        left, right = states[0], states[1]

    if left.encrypted and not right.encrypted:
        return _mul_plain(left, right, node, ckks)
    if right.encrypted and not left.encrypted:
        return _mul_plain(right, left, node, ckks)
    if not left.encrypted and not right.encrypted:
        poly_degree = max(2, _as_int(ckks.get("poly_degree"), 32768))
        sigma0 = math.sqrt(
            poly_degree
            * (
                (left.sigma0 * right.message_sigma) ** 2
                + (right.sigma0 * left.message_sigma) ** 2
                + (left.sigma0 * right.sigma0) ** 2
            )
        )
        return _NoiseState(
            sigma0=sigma0,
            sigma1=0.0,
            message_sigma=math.sqrt(poly_degree) * left.message_sigma * right.message_sigma,
            scale_bits=left.scale_bits + right.scale_bits,
            level=min(left.level, right.level),
            encrypted=False,
            node=node,
        )
    return _mul_cipher(left, right, node, ckks)


def _mul_plain(cipher: _NoiseState, plain: _NoiseState, node: str, ckks: dict[str, Any]) -> _NoiseState:
    poly_degree = max(2, _as_int(ckks.get("poly_degree"), 32768))
    # The q/scaling factor is already represented in plain.message_sigma.
    plain_energy = plain.message_sigma * plain.message_sigma + plain.sigma0 * plain.sigma0
    return _NoiseState(
        sigma0=math.sqrt(poly_degree * cipher.sigma0 * cipher.sigma0 * plain_energy),
        sigma1=math.sqrt(poly_degree * cipher.sigma1 * cipher.sigma1 * plain_energy),
        message_sigma=math.sqrt(poly_degree) * cipher.message_sigma * plain.message_sigma,
        scale_bits=cipher.scale_bits + plain.scale_bits,
        level=min(cipher.level, plain.level),
        encrypted=True,
        node=node,
    )


def _mul_cipher(left: _NoiseState, right: _NoiseState, node: str, ckks: dict[str, Any]) -> _NoiseState:
    poly_degree = max(2, _as_int(ckks.get("poly_degree"), 32768))
    t0 = math.sqrt(
        poly_degree
        * (
            (left.message_sigma * right.sigma0) ** 2
            + (right.message_sigma * left.sigma0) ** 2
            + (left.sigma0 * right.sigma0) ** 2
        )
    )
    t1 = math.sqrt(
        poly_degree
        * (
            left.sigma1 * left.sigma1 * (right.message_sigma * right.message_sigma + right.sigma1 * right.sigma1)
            + right.sigma1 * right.sigma1 * (left.message_sigma * left.message_sigma + left.sigma1 * left.sigma1)
        )
    )
    t2 = math.sqrt(poly_degree) * left.sigma1 * right.sigma1
    relin0, relin1 = _keyswitch_added_sigma(t2, ckks, squared_secret=True)
    return _NoiseState(
        sigma0=_hypot(t0, relin0),
        sigma1=_hypot(t1, relin1),
        message_sigma=math.sqrt(poly_degree) * left.message_sigma * right.message_sigma,
        scale_bits=left.scale_bits + right.scale_bits,
        level=min(left.level, right.level),
        encrypted=True,
        node=node,
    )


def _rotate_state(state: _NoiseState, ckks: dict[str, Any], node: str) -> _NoiseState:
    add0, add1 = _keyswitch_added_sigma(state.sigma1, ckks, squared_secret=False)
    return _NoiseState(
        sigma0=_hypot(state.sigma0, add0),
        sigma1=_hypot(state.sigma1, add1),
        message_sigma=state.message_sigma,
        scale_bits=state.scale_bits,
        level=state.level,
        encrypted=state.encrypted,
        node=node,
    )


def _upscale_state(state: _NoiseState, bits: int, node: str) -> _NoiseState:
    factor = _pow2(bits)
    return _NoiseState(
        sigma0=state.sigma0 * factor,
        sigma1=state.sigma1 * factor,
        message_sigma=state.message_sigma * factor,
        scale_bits=state.scale_bits + bits,
        level=state.level,
        encrypted=state.encrypted,
        node=node,
    )


def _bootstrap_state(state: _NoiseState, ckks: dict[str, Any], node: str) -> _NoiseState:
    scale_bits = max(1, _as_int(ckks.get("Sf"), state.scale_bits))
    level = max(_as_int(ckks.get("bts_ub"), state.level), _as_int(ckks.get("lvl_ub"), state.level))
    input_bound = _coefficient_bound_sigma(state, ckks)
    # Bootstrapping is a circuit refresh. Without the full Lattigo bootstrap plan in
    # the Python request, we use a conservative mathematical envelope: fresh public-key
    # noise plus one decoded input-error carry term and one rescale rounding term.
    carry = input_bound / max(1.0, _pow2(max(0, state.scale_bits - scale_bits)))
    sigma0 = _hypot(math.sqrt(PUBLIC_KEY_FRESH_VARIANCE_0), carry, math.sqrt(ROUNDING_VARIANCE))
    sigma1 = math.sqrt(PUBLIC_KEY_FRESH_VARIANCE_1)
    return _NoiseState(
        sigma0=sigma0,
        sigma1=sigma1,
        message_sigma=max(state.message_sigma / max(1.0, _pow2(max(0, state.scale_bits - scale_bits))), DEFAULT_INPUT_STD * _pow2(scale_bits)),
        scale_bits=scale_bits,
        level=level,
        encrypted=True,
        node=node,
    )


def _apply_transition(
    state: _NoiseState,
    out_level: int,
    out_scale: int,
    ckks: dict[str, Any],
    node: str,
) -> _NoiseState:
    sf_bits = max(1, _as_int(ckks.get("Sf"), 40))
    result = state
    if not _check_res(sf_bits, state.level, state.scale_bits, out_level, out_scale):
        result = _bootstrap_state(result, ckks, node)

    while result.scale_bits - out_scale > 0:
        step_bits = min(sf_bits, result.scale_bits - out_scale)
        result = _rescale_once(result, step_bits, node)

    if out_scale > result.scale_bits:
        result = _upscale_state(result, out_scale - result.scale_bits, node)

    return _NoiseState(
        sigma0=result.sigma0,
        sigma1=result.sigma1,
        message_sigma=result.message_sigma,
        scale_bits=out_scale,
        level=out_level,
        encrypted=result.encrypted,
        node=result.node,
    )


def _rescale_once(state: _NoiseState, factor_bits: int, node: str) -> _NoiseState:
    factor = _pow2(max(0, factor_bits))
    return _NoiseState(
        sigma0=_hypot(state.sigma0 / factor, math.sqrt(ROUNDING_VARIANCE)),
        sigma1=_hypot(state.sigma1 / factor, math.sqrt(ROUNDING_VARIANCE)),
        message_sigma=state.message_sigma / factor,
        scale_bits=state.scale_bits - factor_bits,
        level=max(0, state.level - 1),
        encrypted=state.encrypted,
        node=node,
    )


def _keyswitch_added_sigma(
    switched_component_sigma: float,
    ckks: dict[str, Any],
    *,
    squared_secret: bool,
) -> tuple[float, float]:
    poly_degree = max(2, _as_int(ckks.get("poly_degree"), 32768))
    secret_sigma = _secret_component_sigma(poly_degree, ckks, squared=squared_secret)
    fresh_sigma = max(
        0.0,
        _as_float(ckks.get("fresh_encryption_sigma"), DEFAULT_FRESH_ENCRYPTION_SIGMA),
    )
    # This is the key-switch addend in Lemma 13 with the RNS decomposition/P term
    # collapsed to a conservative one-prime Gaussian envelope.
    eval_key_sigma_sq = poly_degree * fresh_sigma * fresh_sigma * ROUNDING_VARIANCE
    switched_sigma_sq = poly_degree * switched_component_sigma * switched_component_sigma * secret_sigma * secret_sigma
    return math.sqrt(switched_sigma_sq + eval_key_sigma_sq + ROUNDING_VARIANCE), math.sqrt(ROUNDING_VARIANCE)


def _decoded_bound_bits(
    state: _NoiseState,
    ckks: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    alpha = max(0.0, _as_float(policy.get("alpha"), DEFAULT_ALPHA))
    coefficient_sigma = _coefficient_bound_sigma(state, ckks)
    coefficient_bound = max(alpha * coefficient_sigma, 2.0 ** -1022)
    estimated_noise_bits = math.log2(coefficient_bound)
    estimated_precision_bits = max(0.0, float(state.scale_bits) - estimated_noise_bits)
    return {
        "node": state.node,
        "level": state.level,
        "scale_bits": state.scale_bits,
        "sigma0": state.sigma0,
        "sigma1": state.sigma1,
        "coefficient_sigma": coefficient_sigma,
        "alpha": alpha,
        "coefficient_bound": coefficient_bound,
        "estimated_noise_bits": estimated_noise_bits,
        "estimated_precision_bits": estimated_precision_bits,
    }


def _coefficient_bound_sigma(state: _NoiseState, ckks: dict[str, Any]) -> float:
    poly_degree = max(2, _as_int(ckks.get("poly_degree"), 32768))
    secret_sigma = _secret_component_sigma(poly_degree, ckks, squared=False)
    return math.sqrt(state.sigma0 * state.sigma0 + poly_degree * state.sigma1 * state.sigma1 * secret_sigma * secret_sigma)


def _secret_component_sigma(poly_degree: int, ckks: dict[str, Any], *, squared: bool) -> float:
    hamming = _as_int(ckks.get("secret_hamming_weight"), 0)
    if hamming <= 0:
        hamming = min(192, poly_degree)
    hamming = max(1, min(poly_degree, hamming))
    sigma = math.sqrt(hamming / poly_degree)
    if not squared:
        return sigma
    return max(sigma * sigma * math.sqrt(poly_degree), sigma)


def _bound_model(ckks: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    alpha = max(0.0, _as_float(policy.get("alpha"), DEFAULT_ALPHA))
    poly_degree = max(2, _as_int(ckks.get("poly_degree"), 32768))
    tail = math.erf(alpha / math.sqrt(2.0)) if alpha > 0.0 else 0.0
    if tail <= 0.0:
        success = 0.0
        failure = 1.0
    elif tail >= 1.0:
        success = 1.0
        failure = 0.0
    else:
        log_success = poly_degree * math.log(tail)
        success = math.exp(log_success) if log_success > -745.0 else 0.0
        failure = -math.expm1(log_success) if log_success > -1.0 else 1.0 - success
    return {
        "alpha": alpha,
        "tail_success_probability": success,
        "tail_failure_probability": max(0.0, min(1.0, failure)),
        "probability_formula": "erf(alpha/sqrt(2))^N",
        "decoded_bound_formula": "alpha * sqrt(sigma_0^2 + N*sigma_1^2*sigma_s^2) / Delta",
        "component_noise": "coefficient-domain sigma per ciphertext component",
    }


def _noise_hotspots(
    trace: dict[str, dict[str, Any]],
    ckks: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    ranked = sorted(
        trace.items(),
        key=lambda item: _decoded_bound_bits(item[1]["state"], ckks, policy)["estimated_noise_bits"],
        reverse=True,
    )
    return [
        {
            "location": f"node={node};op={entry['op']}",
            "estimated_noise_bits": _decoded_bound_bits(entry["state"], ckks, policy)["estimated_noise_bits"],
        }
        for node, entry in ranked[:8]
    ]


def _trace_entry(op: str, state: _NoiseState) -> dict[str, Any]:
    return {"op": op, "state": state}


def _check_res(sf_bits: int, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
    if in_lvl < out_lvl:
        return False
    return sf_bits * in_lvl - in_scl >= sf_bits * out_lvl - out_scl


def _map_int(assignment: dict[str, Any], section: str, key: str, default: int) -> int:
    return _as_int((assignment.get(section) or {}).get(str(key)), default)


def _edge_key(u: str, v: str) -> str:
    return f"{u}->{v}"


def _pow2(bits: int | float) -> float:
    bits = max(-1022.0, min(1023.0, float(bits)))
    return 2.0 ** bits


def _hypot(*values: float) -> float:
    return math.hypot(*(float(value) for value in values))


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
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if math.isfinite(result) else float(default)
