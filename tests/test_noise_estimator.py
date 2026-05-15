from __future__ import annotations

from types import SimpleNamespace

from scripts.optimizer.orbit.noise_estimator import estimate_compile_result_noise


def _params(**overrides):
    data = {
        "noise_estimator_binary": None,
        "noise_estimator_timeout_sec": 1,
        "noise_estimator_min_output_margin_bits": 2.0,
        "noise_estimator_alpha": 14.0,
        "poly_deg": 32768,
        "max_slot": 16384,
        "Sf": 40,
        "Sw": 40,
        "lvl_lb": 1,
        "lvl_ub": 16,
        "bts_lb": 3,
        "bts_ub": 16,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _single_input_context():
    return {
        "ckks": {
            "poly_degree": 32768,
            "max_slots": 16384,
            "Sf": 40,
            "Sw": 40,
            "lvl_lb": 1,
            "lvl_ub": 16,
            "bts_lb": 3,
            "bts_ub": 16,
        },
        "tdag": {
            "inputs": ["arg0"],
            "outputs": ["arg0"],
            "topological_order": ["arg0"],
            "nodes": {
                "arg0": {
                    "op": "input",
                    "weight": 1,
                    "op_descr": {},
                    "comment": "",
                }
            },
            "edges": [],
        },
    }


def _single_input_result():
    return {
        "valid": True,
        "bootstrap_count": 0,
        "rescale_count": 0,
        "fallback_selected_budgets": 0,
        "profile_risk": 0.0,
        "selected_output_state": {"out_scl": 40},
        "assignment": {
            "v_lvl_in": {"arg0": 16},
            "v_scl_in": {"arg0": 40},
            "v_lvl_out": {"arg0": 16},
            "v_scl_out": {"arg0": 40},
            "e_lvl_out": {},
            "e_scl_out": {},
        },
    }


def test_noise_estimator_replays_assignment_with_componentwise_bound():
    estimate = estimate_compile_result_noise(
        _single_input_context(),
        _single_input_result(),
        _params(),
    )

    assert estimate["mode"] == "internal_componentwise_average_case"
    assert estimate["model"] == "componentwise-average-case-ckks"
    assert estimate["valid"] is True
    assert estimate["estimated_precision_bits"] > 30.0
    assert estimate["bound_model"]["decoded_bound_formula"].startswith("alpha * sqrt")
    assert estimate["details"]["assignment_replayed"] is True


def test_noise_estimator_alpha_changes_mathematical_tail_bound():
    context = _single_input_context()
    result = _single_input_result()

    conservative = estimate_compile_result_noise(context, result, _params(noise_estimator_alpha=14.0))
    tighter = estimate_compile_result_noise(context, result, _params(noise_estimator_alpha=6.0))

    assert conservative["bound_model"]["alpha"] == 14.0
    assert tighter["bound_model"]["alpha"] == 6.0
    assert tighter["estimated_precision_bits"] > conservative["estimated_precision_bits"]
    assert tighter["bound_model"]["tail_failure_probability"] > conservative["bound_model"]["tail_failure_probability"]


def test_noise_estimator_aggregate_bound_is_mathematical_fallback():
    estimate = estimate_compile_result_noise(
        {"ckks": {"poly_degree": 32768, "Sf": 40, "Sw": 40, "lvl_ub": 16}},
        {
            "bootstrap_count": 0,
            "rescale_count": 2,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
        },
        _params(),
    )

    assert estimate["mode"] == "internal_componentwise_aggregate_bound"
    assert estimate["valid"] is True
    assert estimate["details"]["assignment_replayed"] is False
