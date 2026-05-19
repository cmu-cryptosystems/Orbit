"""Params loading and ilp_solver validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.params.params import Params


def test_params_toy_runtime(toy_cost_json: str):
    p = Params(toy_cost_json, "Orbit", "compile")
    assert p.backend == "Toy"
    assert p.placement_backend == "openevolve"
    assert p.ilp_solver == "pulp"
    assert p.openevolve_provider == "gemini"
    assert p.openevolve_model == "gemini-3.1-flash-lite"
    assert p.openevolve_api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert p.openevolve_api_key_env == "OPENAI_API_KEY"
    assert p.openevolve_primary_weight == 1.0
    assert p.openevolve_secondary_provider == "none"
    assert p.openevolve_secondary_model == "gpt-5.5"
    assert p.openevolve_secondary_api_base is None
    assert p.openevolve_secondary_api_key_env == "OPENAI_API_KEY"
    assert p.openevolve_secondary_weight == 0.25
    assert p.openevolve_harness == "compile"
    assert p.openevolve_search_mode == "bootstrap-mcts"
    assert p.openevolve_granularity == "layer-nonlinear"
    assert p.openevolve_leniency == "repair"
    assert p.openevolve_max_unit_samples == 64
    assert p.openevolve_eval_suite == "polybert-sampled"
    assert p.openevolve_reference_json is None
    assert p.openevolve_finalists == 3
    assert p.openevolve_budget_aggressive is True
    assert p.openevolve_target_bootstrap_count == 0
    assert p.openevolve_llm_timeout_sec == 180
    assert p.openevolve_llm_retries == 1
    assert p.openevolve_llm_retry_delay_sec == 2
    assert p.openevolve_evaluator_timeout_sec == 180
    assert p.openevolve_parallel_evaluations == 1
    assert p.openevolve_checkpoint_interval == 5
    assert p.openevolve_fail_open is True
    assert p.noise_estimator == "finalists"
    assert p.noise_estimator_binary is None
    assert p.noise_estimator_timeout_sec == 30
    assert p.noise_estimator_min_output_margin_bits == 2.0
    assert p.noise_estimator_alpha == 14.0
    assert p.noise_estimator_max_trace_message_bits == 20.0
    assert p.noise_estimator_require_trace_safe is False
    assert p.mode == "compile"


def test_params_ilp_solver_override(toy_cost_json: str):
    p = Params(toy_cost_json, "Orbit", "compile", ilp_solver="pulp")
    assert p.ilp_solver == "pulp"


def test_params_ilp_solver_from_json(tmp_path: Path, toy_cost_json: str):
    with open(toy_cost_json) as f:
        data = json.load(f)
    data["ilp_solver"] = "pulp"
    jf = tmp_path / "cfg.json"
    with open(jf, "w") as f:
        json.dump(data, f)
    p = Params(str(jf), "Orbit", "compile")
    assert p.ilp_solver == "pulp"


def test_params_rejects_bad_ilp_solver(toy_cost_json: str):
    with pytest.raises(ValueError, match="ilp_solver"):
        Params(toy_cost_json, "Orbit", "compile", ilp_solver="not_a_solver")


def test_params_rejects_bad_openevolve_provider(toy_cost_json: str):
    with pytest.raises(ValueError, match="openevolve_provider"):
        Params(toy_cost_json, "Orbit", "compile", openevolve_provider="not_a_provider")


def test_params_rejects_bad_openevolve_secondary_provider(toy_cost_json: str):
    with pytest.raises(ValueError, match="openevolve_secondary_provider"):
        Params(toy_cost_json, "Orbit", "compile", openevolve_secondary_provider="bogus")


def test_params_rejects_bad_openevolve_search_mode(toy_cost_json: str):
    with pytest.raises(ValueError, match="openevolve_search_mode"):
        Params(toy_cost_json, "Orbit", "compile", openevolve_search_mode="bogus")


def test_params_rejects_bad_openevolve_harness(toy_cost_json: str):
    with pytest.raises(ValueError, match="openevolve_harness"):
        Params(toy_cost_json, "Orbit", "compile", openevolve_harness="batch")


def test_params_rejects_bad_openevolve_granularity(toy_cost_json: str):
    with pytest.raises(ValueError, match="openevolve_granularity"):
        Params(toy_cost_json, "Orbit", "compile", openevolve_granularity="node")


def test_params_rejects_bad_openevolve_leniency(toy_cost_json: str):
    with pytest.raises(ValueError, match="openevolve_leniency"):
        Params(toy_cost_json, "Orbit", "compile", openevolve_leniency="unsafe")


def test_params_rejects_bad_openevolve_eval_suite(toy_cost_json: str):
    with pytest.raises(ValueError, match="openevolve_eval_suite"):
        Params(toy_cost_json, "Orbit", "compile", openevolve_eval_suite="large")


def test_params_rejects_bad_noise_estimator(toy_cost_json: str):
    with pytest.raises(ValueError, match="noise_estimator"):
        Params(toy_cost_json, "Orbit", "compile", noise_estimator="always")


def test_relax_only_lowers_default_constant_scale(
    toy_cost_json: str, tmp_path: Path
):
    profile = {
        "schema_version": "orbit-resilience-constraints-v0",
        "constraints": [{"node": "0", "min_scale": 16, "ports": ["in", "out"]}],
    }
    profile_path = tmp_path / "constraints.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    p = Params(
        toy_cost_json,
        "Orbit",
        "compile",
        Sw=40,
        CSw=None,
        resilience_profile=str(profile_path),
        resilience_constraint_policy="relax-only",
    )

    assert p.Sw == 16
    assert p.Csw == 16


def test_explicit_constant_scale_is_respected_under_relax_only(
    toy_cost_json: str, tmp_path: Path
):
    profile = {
        "schema_version": "orbit-resilience-constraints-v0",
        "constraints": [{"node": "0", "min_scale": 16, "ports": ["in", "out"]}],
    }
    profile_path = tmp_path / "constraints.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    p = Params(
        toy_cost_json,
        "Orbit",
        "compile",
        Sw=40,
        CSw=24,
        resilience_profile=str(profile_path),
        resilience_constraint_policy="relax-only",
    )

    assert p.Sw == 16
    assert p.Csw == 24
