from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.assignment import Assign
from scripts.latency_estimator.latency_estimator import LatencyEstimator
import scripts.optimizer.orbit.openevolve_backend as oe_backend
from scripts.optimizer.orbit.openevolve_backend import (
    OpenEvolvePlacementWorker,
    PlacementBuilder,
    PlacementConstraints,
    PlacementMCTS,
    _aggregate_counts,
    build_context,
    build_compile_context,
    candidate_actions,
    evaluate_compile_candidate_program,
    evaluate_candidate_program,
    run_compile_openevolve,
    serialize_assignment_record,
    solve_budget_batch,
    tdag_from_context,
    transition_slack,
    valid_transition,
)
from scripts.optimizer.orbit.qbp_manager import QBPManager
from scripts.optimizer.orbit.orbit_core import orbit_core
from scripts.params.params import Params
from scripts.tdag.tdag import Tdag


def _params(toy_cost_json: str, **kwargs) -> Params:
    return Params(
        toy_cost_json,
        "Orbit",
        "compile",
        Sw=40,
        CSw=40,
        threads=1,
        comp=False,
        part=False,
        placement_backend="openevolve",
        **kwargs,
    )


def _toy_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_toy")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    graph.add_node(
        "0",
        op="mul",
        weight=1,
        level=None,
        scale=None,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.attention.self.query;op=linear",
    )
    graph.add_edge("arg0", "0", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"0"}
    return graph


def _mul_chain_pdag(params: Params, length: int = 8) -> Tdag:
    graph = Tdag(params, "oe_mul_chain")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    prev = "arg0"
    for idx in range(length):
        node = f"mul{idx}"
        graph.add_node(
            node,
            op="mul",
            weight=1,
            level=None,
            scale=None,
            op_descr={"single": 1, "double": 0},
            comment=f"scope=chain.{idx};op=mul",
        )
        graph.add_edge(prev, node, weight=1)
        prev = node
    graph.inputs = {"arg0"}
    graph.outputs = {prev}
    return graph


def _branch_merge_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_branch_merge")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    for name in ("left", "right"):
        graph.add_node(
            name,
            op="mul",
            weight=1,
            level=None,
            scale=None,
            op_descr={"single": 1, "double": 0},
            comment=f"scope={name};op=mul",
        )
        graph.add_edge("arg0", name, weight=1)
    graph.add_node("merge", op="add", weight=1, level=None, scale=None, op_descr={"single": 1}, comment="")
    graph.add_edge("left", "merge", weight=1)
    graph.add_edge("right", "merge", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"merge"}
    return graph


def _constant_bias_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_constant_bias")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    graph.add_node(
        "bias",
        op="constant",
        weight=1,
        level=None,
        scale=None,
        op_descr={},
        comment="scope=classifier.bias;op=bias",
    )
    graph.add_node(
        "add",
        op="add",
        weight=1,
        level=None,
        scale=None,
        op_descr={"single": 1},
        comment="scope=classifier;op=linear_bias",
    )
    graph.add_edge("arg0", "add", weight=1)
    graph.add_edge("bias", "add", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"add"}
    return graph


def _constant_mul_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_constant_mul")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    graph.add_node(
        "weight",
        op="constant",
        weight=1,
        level=None,
        scale=None,
        op_descr={},
        comment="scope=dense.weight;op=weight",
    )
    graph.add_node(
        "mul",
        op="mul",
        weight=1,
        level=None,
        scale=None,
        op_descr={"single": 0, "double": 1},
        comment="scope=dense;op=linear",
    )
    graph.add_edge("arg0", "mul", weight=1)
    graph.add_edge("weight", "mul", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"mul"}
    return graph


def _relaxing_profile_path(tmp_path: Path) -> str:
    profile = {
        "schema_version": "orbit-resilience-constraints-v0",
        "constraints": [{"node": "add", "min_scale": 16, "ports": ["in", "out"]}],
    }
    profile_path = tmp_path / "relaxing_constraints.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return str(profile_path)


class FakeLLMModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLMConfig:
    def __init__(self):
        self.api_base = None
        self.models = []
        self.evaluator_models = []
        self.timeout = None
        self.retries = None
        self.retry_delay = None
        self.max_tokens = None
        self.temperature = None

    def update_model_params(self, args, overwrite=False):
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


class FakeDatabaseConfig:
    def __init__(self):
        self.random_seed = None
        self.feature_dimensions = []
        self.log_prompts = None
        self.num_islands = 5


class FakeEvaluatorConfig:
    def __init__(self):
        self.timeout = None
        self.parallel_evaluations = None
        self.max_retries = None


class FakePromptConfig:
    def __init__(self):
        self.max_artifact_bytes = 20 * 1024


class FakeConfig:
    def __init__(self):
        self.random_seed = None
        self.checkpoint_interval = None
        self.diff_based_evolution = True
        self.max_code_length = 10_000
        self.llm = FakeLLMConfig()
        self.database = FakeDatabaseConfig()
        self.evaluator = FakeEvaluatorConfig()
        self.prompt = FakePromptConfig()


def _install_fake_openevolve(monkeypatch, fake_run_evolution, loaded_config=None):
    fake_openevolve = types.ModuleType("openevolve")
    fake_openevolve.__path__ = []
    fake_openevolve.run_evolution = fake_run_evolution

    fake_config_mod = types.ModuleType("openevolve.config")
    fake_config_mod.Config = FakeConfig
    fake_config_mod.LLMModelConfig = FakeLLMModelConfig
    fake_config_mod.load_config = lambda _path: loaded_config or FakeConfig()
    fake_openevolve.config = fake_config_mod

    monkeypatch.setitem(sys.modules, "openevolve", fake_openevolve)
    monkeypatch.setitem(sys.modules, "openevolve.config", fake_config_mod)


def test_context_roundtrip_and_conservative_placement(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    budgets = [{"in_lvl": -1, "in_scl": 40}]

    context = build_context(graph, budgets, params)
    restored = tdag_from_context(context)

    assert restored.name == graph.name
    assert restored.nodes["0"]["op"] == "mul"
    assert context["schema_version"] == "orbit-openevolve-placement-context-v4"
    assert context["tdag"]["topological_order"]
    assert context["graph_summary"]["max_scale"] == params.Sf + 2 * params.Sw
    assert context["budget_summary"]["count"] == 1
    assert context["constraints"]["local_scale_lower_bounds"]["0"]["in"] == 40
    assert context["constraints"]["ilp_semantics"]["solver_free"] is True
    assert "decryptable" in context["constraints"]["ilp_semantics"]["constraints"]
    assert context["latency_model"]["available_ops"]

    le = LatencyEstimator(params)
    io_to_assign, io_to_cost = solve_budget_batch(graph, budgets, le, params)

    assert io_to_assign
    assert io_to_cost
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True


def test_level_preserving_scheduler_limits_mul_chain_bootstraps(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=8)
    le = LatencyEstimator(params)

    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving", "refresh_fanout_at_level_floor": True},
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    counts = _aggregate_counts([assign])
    assert assign.check_assign() is True
    assert counts["bootstrap"] < 8


def test_level_preserving_fallback_keeps_batch_complete_and_tracks_improvement(
    toy_cost_json: str,
):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=4)
    le = LatencyEstimator(params)
    diagnostics = {}

    _io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving", "allow_seed_fallback": True},
        diagnostics,
    )

    _default_assign, default_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
    )

    assert diagnostics["solved_budgets"] == 1
    assert diagnostics["candidate_solved_budgets"] == 1
    assert diagnostics["fallback_solved_budgets"] == 1
    assert diagnostics["fallback_selected_budgets"] == 0
    assert diagnostics["candidate_improved_budgets"] == 0
    assert min(next(iter(io_to_cost.values())).values()) == min(
        next(iter(default_cost.values())).values()
    )


def test_seed_fallback_does_not_reward_invalid_candidate(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    seed_program = tmp_path / "seed.py"
    invalid_program = tmp_path / "invalid.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    seed_program.write_text(
        "def place(context):\n    return {'strategy': 'waterline_seed'}\n",
        encoding="utf-8",
    )
    invalid_program.write_text(
        "def place(context):\n"
        "    return {\n"
        "        'preferred_node_scales': {'0': 9999},\n"
        "        'allow_seed_fallback': True,\n"
        "    }\n",
        encoding="utf-8",
    )

    seed = evaluate_candidate_program(context_path, seed_program)
    invalid = evaluate_candidate_program(context_path, invalid_program)

    assert invalid["metrics"]["validity"] == 0.0
    assert invalid["metrics"]["effective_validity"] == 1.0
    assert invalid["metrics"]["fallback_selected_budgets"] == 1.0
    assert invalid["metrics"]["combined_score"] > 0.0
    assert invalid["metrics"]["combined_score"] < seed["metrics"]["combined_score"]


def test_level_preserving_scheduler_handles_branch_merge(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving"},
    )

    assert io_to_assign
    assert io_to_cost
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True
    assert assign.v_scl_in["merge"] >= params.scale_lower_bound("merge", graph.nodes["merge"], "in")


def test_level_preserving_scheduler_handles_bypass_budget(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)
    budget = {
        "in_lvl": -1,
        "in_scl": 40,
        "maino_v": "left",
        "main_dag_size": 3,
        "main_qbp_cost": {(12, 80): 10.0, (10, 40): 20.0},
    }

    io_to_assign, io_to_cost = solve_budget_batch(graph, [budget], le, params)

    assert io_to_assign
    assert io_to_cost
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True
    assert (assign.v_lvl_in["left"], assign.v_scl_in["left"]) in budget["main_qbp_cost"]


def test_relax_only_respects_explicit_constant_scale_for_additive_constants(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(
        toy_cost_json,
        resilience_profile=_relaxing_profile_path(tmp_path),
        resilience_constraint_policy="relax-only",
    )
    graph = _constant_bias_pdag(params)
    le = LatencyEstimator(params)

    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 16}],
        le,
        params,
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert params.Sw == 16
    assert params.Csw == 40
    assert assign.check_assign() is True
    assert assign.v_scl_out["bias"] == params.Csw
    assert assign.v_scl_in["add"] >= params.Csw


def test_level_preserving_allows_relaxed_waterline_constant_mul(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(
        toy_cost_json,
        resilience_profile=_relaxing_profile_path(tmp_path),
        resilience_constraint_policy="relax-only",
    )
    graph = _constant_mul_pdag(params)
    le = LatencyEstimator(params)
    diagnostics = {}

    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving", "allow_seed_fallback": True},
        diagnostics,
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert params.Sw == 16
    assert params.Csw == 40
    assert build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)["graph_summary"][
        "max_scale"
    ] == params.Sf + 2 * params.Csw
    assert diagnostics["candidate_solved_budgets"] == 1
    assert assign.check_assign() is True
    assert assign.v_scl_in["mul"] == 80


def test_evaluate_candidate_program_returns_metrics_and_artifacts(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "candidate.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    program_path.write_text(
        "def place(context):\n"
        "    return {'preferred_node_scales': {'0': 40}}\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert result["metrics"]["combined_score"] > 0.0
    assert result["metrics"]["graph_nodes"] == 2.0
    assert "invalid_reasons" in result["artifacts"]
    assert "reference_avg_latency_usec" in result["artifacts"]
    assert "policy_summary" in result["artifacts"]


def test_builder_api_explicit_placement_record(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budget = {"in_lvl": -1, "in_scl": 40}
    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [budget],
        le,
        params,
        {"strategy": "level_preserving"},
    )
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    record = serialize_assignment_record(graph.name, budget, assign)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "candidate.py"
    context_path.write_text(json.dumps(build_context(graph, [budget], params)), encoding="utf-8")
    program_path.write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        f"RECORD = {record!r}\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).placement_records([RECORD])\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert result["metrics"]["fallback_selected_budgets"] == 0.0
    assert result["metrics"]["combined_score"] > 0.0


def test_open_helpers_expose_ilp_compatible_transition(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": -1, "in_scl": 40}], params)

    assert valid_transition(context, 16, 40, 15, 40)
    assert not valid_transition(context, 1, 131, 16, 40)
    assert not valid_transition(context, 16, params.max_scale() + 1, 15, 40)
    slack = transition_slack(context, 16, 40, 15, 40)
    assert slack["valid"] is True
    assert slack["uses_bootstrap"] is False
    assert slack["transition_reserve_bits"] >= 0
    invalid = transition_slack(context, 1, 131, 16, 40)
    assert invalid["valid"] is False
    assert "input_not_decryptable" in invalid["violations"]
    above_max = transition_slack(context, 16, params.max_scale() + 1, 15, 40)
    assert above_max["valid"] is False
    assert "input_scale_above_max" in above_max["violations"]
    assert PlacementBuilder(context).level_preserving()["policy"]["strategy"] == "level_preserving"
    constraints = PlacementConstraints(context)
    assert constraints.decryptability_bound(16) == params.Sf * (16 - params.lvl_lb + 2) - 7
    assert constraints.boundary_output_scale_bound(16) == params.Sf * (16 + 1) - 7
    assert constraints.valid_transition(16, 40, 15, 40)
    assert constraints.transition_slack(16, 40, 15, 40)["valid"] is True
    assert constraints.node_scale_lower_bound("0", "out") == params.Sw
    assert constraints.edge_scale_lower_bound("arg0", "0") == params.Sw
    assert constraints.mul_input_scale([40]) == 80
    states = constraints.legal_output_states(16, 40, include_bootstrap=False)
    assert states
    assert all(not state["uses_bootstrap"] for state in states)
    assert states[0]["level"] <= params.lvl_ub
    assert PlacementBuilder(context).constraints.legal_output_states(16, 40)


def test_tuneinsight_noise_slack_relaxes_reserve_penalty(toy_cost_json: str):
    params = _params(toy_cost_json)
    base_policy = {
        "bootstrap_penalty": 0.0,
        "rescale_penalty": 0.0,
        "level_drop_penalty": 0.0,
        "reserve_penalty": 10.0,
        "min_transition_reserve": 1000,
        "min_decryptability_reserve": 1000,
    }

    worst_score = oe_backend._transition_score(
        params,
        None,
        16,
        40,
        15,
        40,
        {**base_policy, "noise_slack_model": "worst_case"},
    )
    avgcase_score = oe_backend._transition_score(
        params,
        None,
        16,
        40,
        15,
        40,
        {**base_policy, "noise_slack_model": "tuneinsight_avgcase"},
    )
    ignored_score = oe_backend._transition_score(
        params,
        None,
        16,
        40,
        15,
        40,
        {**base_policy, "noise_slack_model": "off"},
    )

    assert worst_score is not None
    assert avgcase_score is not None
    assert ignored_score is not None
    assert 0.0 == ignored_score < avgcase_score < worst_score


def test_assignment_validation_rejects_non_decryptable_candidate_record(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    assign = Assign(graph)
    too_large_scale = params.max_scale() + 1
    assign.v_lvl_in["arg0"] = params.lvl_ub
    assign.v_scl_in["arg0"] = 40
    assign.v_lvl_out["arg0"] = params.lvl_ub
    assign.v_scl_out["arg0"] = 40
    assign.e_lvl_out[("arg0", "0")] = params.lvl_ub
    assign.e_scl_out[("arg0", "0")] = 40
    assign.v_lvl_out["0"] = params.lvl_ub
    assign.v_scl_out["0"] = too_large_scale

    with pytest.raises(ValueError, match="decryptability bounds"):
        assign.check_assign()


def test_none_min_internal_level_matches_ilp_lower_bound(toy_cost_json: str):
    params = _params(toy_cost_json)
    options = oe_backend._policy_options({"strategy": "level_preserving", "min_internal_level": None}, params)

    assert options["min_internal_level"] is None
    assert oe_backend._effective_min_internal_level(options, params) == params.lvl_lb


def test_patch_vocabulary_and_portfolio_normalize(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "candidate.py"
    program_path.write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    builder = PlacementBuilder(context)\n"
        "    builder.set_policy('strategy', 'latency_beam').prefer_node_level('0', 16).disable_seed_fallback()\n"
        "    return builder.portfolio(builder.latency_beam()['policy'], builder.level_preserving()['policy'])\n",
        encoding="utf-8",
    )

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["strategy"] == "latency_beam"
    assert hints["allow_seed_fallback"] is False
    assert hints["preferred_node_levels"]["0"] == 16
    assert len(hints["portfolio"]) == 2
    assert oe_backend._static_validate_hints(context, hints)["valid"] is True


def test_latency_beam_policy_produces_valid_assignment(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=4)
    le = LatencyEstimator(params)

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "latency_beam", "beam_width": 4},
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True
    assert io_to_cost


def test_builder_depth_fanout_policy_uses_graph_summary(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)

    policy = PlacementBuilder(context).depth_fanout_aware()["policy"]

    assert policy["strategy"] == "level_preserving"
    assert policy["refresh_fanout_at_level_floor"] is True
    assert policy["min_internal_level"] == params.bts_lb + 1
    assert policy["preferred_node_levels"]["mul0"] == params.lvl_ub


def test_builder_noise_guarded_policy_preserves_headroom(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)

    policy = PlacementBuilder(context).noise_guarded_refresh()["policy"]

    assert policy["strategy"] == "level_preserving"
    assert policy["refresh_fanout_at_level_floor"] is True
    assert policy["allow_seed_fallback"] is False
    assert policy["min_internal_level"] == 12
    assert policy["max_scale_candidates"] == 12
    assert policy["reserve_penalty"] > 0
    assert policy["min_transition_reserve"] > 0


def test_profile_layer_refresh_targets_reference_bert_layernorm_comments(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    graph.nodes["left"]["comment"] = (
        "scope=bert.encoder.layer.0.attention.output.LayerNorm;op=layer_norm"
    )
    graph.nodes["right"]["comment"] = (
        "scope=bert.encoder.layer.0.output.LayerNorm;op=layer_norm"
    )
    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    policy = PlacementBuilder(context).profile_layer_refresh(include_attention=True)["policy"]

    assert policy["preferred_node_levels"]["left"] == params.lvl_ub
    assert policy["preferred_node_levels"]["right"] == params.lvl_ub
    assert policy["preferred_node_scales"]["left"] == params.Sf
    assert policy["preferred_node_scales"]["right"] == params.Sf


def test_builder_compile_seed_includes_bounded_reliable_portfolio(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["strategy"] == "level_preserving"
    assert hints["refresh_fanout_at_level_floor"] is True
    assert hints["max_scale_candidates"] == 10
    assert hints["min_internal_level"] is None
    assert hints["level_drop_penalty"] == 20_000_000.0
    assert hints["allow_seed_fallback"] is False
    assert [policy["strategy"] for policy in hints["portfolio"]] == [
        "latency_beam",
        "latency_beam",
        "level_preserving",
        "latency_beam",
        "level_preserving",
        "level_preserving",
        "level_preserving",
        "level_preserving",
        "level_preserving",
    ]
    assert hints["portfolio"][0]["budget_aggressive"] is True
    assert hints["portfolio"][1]["selection_bootstrap_penalty"] == 500_000_000.0
    assert hints["portfolio"][4]["bootstrap_penalty"] == 650_000_000.0
    assert hints["portfolio"][5]["bootstrap_penalty"] == 650_000_000.0
    assert hints["portfolio"][6]["min_internal_level"] == 6
    assert hints["portfolio"][7]["min_internal_level"] == 8
    assert hints["portfolio"][8]["min_internal_level"] == 12


def test_initial_compile_seed_exposes_active_bootstrap_mcts_knobs(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(
        toy_cost_json,
        openevolve_search_mode="bootstrap-mcts",
        openevolve_budget_aggressive=True,
        openevolve_target_bootstrap_count=9,
    )
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    program_path = tmp_path / "initial.py"
    source = oe_backend._initial_compile_program_source("bootstrap-mcts")
    assert "builder.budget_fulfillment_beam" not in source
    program_path.write_text(source, encoding="utf-8")

    hints = oe_backend._load_candidate_hints(program_path, context)
    sampled = oe_backend._compile_hints_for_eval_suite(hints, "polybert-sampled")

    assert hints["strategy"] == "bootstrap_mcts"
    assert hints["include_seed_repair_actions"] is True
    assert hints["mcts_rollout_budget"] == 24
    assert hints["mcts_action_cap"] == 6
    assert hints["mcts_action_allowlist"] == [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "dense_boundary_cost_beam",
        "minimal_bootstrap_repair",
    ]
    assert hints["mcts_exploration_weight"] == 1.25
    assert hints["mcts_max_repair_bootstraps"] == 16
    assert hints["boundary_group_policies"] == []
    raw_budget_beams = [
        action for action in hints["mcts_actions"] if action.get("name") == "budget_fulfillment_beam"
    ]
    assert raw_budget_beams
    assert raw_budget_beams[0]["prior"] == 0.54
    wide_beams = [
        action for action in hints["mcts_actions"] if action.get("name") == "wide_boundary_cost_beam"
    ]
    assert wide_beams
    presets = hints["mcts_action_presets"]
    assert presets["budget_fulfillment_beam"]["prior"] == 0.65
    assert presets["budget_fulfillment_beam"]["policy"]["beam_width"] == 12
    assert presets["budget_fulfillment_beam"]["policy"]["boundary_state_cap"] == 8
    assert presets["wide_boundary_cost_beam"]["policy"]["boundary_state_cap"] == 10
    assert presets["wide_boundary_cost_beam"]["policy"]["max_scale_candidates"] == 64
    assert presets["dense_boundary_cost_beam"]["policy"]["boundary_state_cap"] == 20
    assert presets["minimal_bootstrap_repair"]["policy"]["selection_objective"] == "min_bootstrap"
    capped_names = [
        action["name"]
        for action in sampled["mcts_actions"][: sampled["mcts_action_cap"]]
    ]
    assert "budget_fulfillment_beam" in capped_names
    assert sampled["mcts_action_cap"] <= 6
    assert set(hints["mcts_action_presets"]) == {
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "dense_boundary_cost_beam",
        "minimal_bootstrap_repair",
    }


def test_mcts_action_presets_change_effective_action_policy(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    hints = PlacementMCTS(context).low_bootstrap_seed(target_bootstraps=9, action_cap=8)
    hints["mcts_action_presets"] = {
        "component_budget_repair": {
            "prior": 0.77,
            "policy": {
                "force_bootstrap_anchors": False,
                "bootstrap_anchor_count": 6,
                "selection_bootstrap_penalty": 12_345.0,
            },
        }
    }

    actions = oe_backend._mcts_actions_from_hints(hints, params)
    component = next(action for action in actions if action.name == "component_budget_repair")

    assert component.prior == 0.77
    assert component.policy["force_bootstrap_anchors"] is False
    assert component.policy["bootstrap_anchor_count"] == 6
    assert component.policy["selection_bootstrap_penalty"] == 12_345.0


def test_mcts_action_priority_filter_changes_action_frontier(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    mcts = PlacementMCTS(context)
    hints = mcts.low_bootstrap_seed(target_bootstraps=9, action_cap=8)
    hints.update(
        mcts.action_focus(
            "component_budget_repair",
            "budget_fulfillment_beam",
            cap=1,
            block=["budget_fulfillment_beam"],
        )
    )
    hints["mcts_action_presets"] = mcts.action_presets(
        component_budget_repair={
            "prior": 0.95,
            "policy": {
                "selection_objective": "component_budget_fit",
                "bootstrap_anchor_count": 5,
            },
        },
        budget_fulfillment_beam={"prior": 0.99, "enabled": False},
    )

    actions = oe_backend._mcts_actions_from_hints(hints, params)

    assert [action.name for action in actions] == ["component_budget_repair"]
    assert actions[0].prior == pytest.approx(0.95)
    assert actions[0].policy["bootstrap_anchor_count"] == 5


def test_policy_effect_summary_detects_seed_equivalent_and_changed_preset(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(
        toy_cost_json,
        openevolve_search_mode="bootstrap-mcts",
        openevolve_budget_aggressive=True,
        openevolve_target_bootstrap_count=9,
    )
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    context["harness"]["eval_suite"] = "polybert-sampled"
    initial_path = tmp_path / "initial_program.py"
    changed_path = tmp_path / "changed.py"
    initial_path.write_text(oe_backend._initial_compile_program_source("bootstrap-mcts"), encoding="utf-8")
    changed_path.write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementMCTS\n"
        "def place(context):\n"
        "    target = 9\n"
        "    mcts = PlacementMCTS(context)\n"
        "    policy = mcts.low_bootstrap_seed(target_bootstraps=target, rollout_budget=12, action_cap=8)\n"
        "    policy['mcts_action_presets'] = mcts.action_presets(\n"
        "        component_budget_repair={'prior': 0.91, 'policy': {'force_bootstrap_anchors': False, 'bootstrap_anchor_count': 6}}\n"
        "    )\n"
        "    return policy\n",
        encoding="utf-8",
    )

    initial_hints = oe_backend._load_candidate_hints(initial_path, context)
    changed_hints = oe_backend._load_candidate_hints(changed_path, context)
    initial_eval = oe_backend._compile_hints_for_eval_suite(initial_hints, "polybert-sampled")
    changed_eval = oe_backend._compile_hints_for_eval_suite(changed_hints, "polybert-sampled")

    seed_effect = oe_backend._policy_effect_summary(context, initial_hints, initial_eval, "polybert-sampled")
    changed_effect = oe_backend._policy_effect_summary(context, changed_hints, changed_eval, "polybert-sampled")

    assert seed_effect["seed_equivalent"] is True
    assert seed_effect["effect_score"] == 0.0
    assert changed_effect["seed_equivalent"] is False
    assert changed_effect["effect_score"] > 0.0
    assert "component_budget_repair" in changed_effect["changed_actions"]


def test_compile_score_does_not_let_seed_equivalent_policy_dominate(
    toy_cost_json: str, tmp_path: Path, monkeypatch
):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_search_mode="bootstrap-mcts",
        openevolve_budget_aggressive=True,
    )
    context = build_compile_context(_branch_merge_pdag(params), params)
    context["reference"] = {
        "final_latency_usec": 100.0,
        "bootstrap_count": 4,
        "rescale_count": 10,
        "valid": False,
    }
    context["harness"]["sampled_seed_baseline"] = {
        "final_latency_usec": 100.0,
        "bootstrap_count": 4,
        "rescale_count": 10,
        "fallback_selected_budgets": 0,
        "fallback_selected_groups": 0,
        "scored_boundary_groups": 2,
        "solved_boundary_groups": 1,
        "candidate_solved_boundary_groups": 1,
    }
    context_path = tmp_path / "compile_context.json"
    seed_program = tmp_path / "initial_program.py"
    changed_program = tmp_path / "changed.py"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    seed_program.write_text("def place(context):\n    return {}\n", encoding="utf-8")
    changed_program.write_text(
        "def place(context):\n    return {'changed_marker': True}\n",
        encoding="utf-8",
    )

    def fake_policy_effect(_context, raw_hints, _eval_hints, _eval_suite):
        changed = bool(raw_hints.get("changed_marker"))
        return {
            "seed_equivalent": not changed,
            "effect_score": 1.0 if changed else 0.0,
            "changed_actions": ["component_budget_repair"] if changed else [],
        }

    def fake_evaluate_compile_hints(_context, _hints, *, suppress_output):
        return {
            "valid": False,
            "validity": 0.0,
            "final_latency_usec": 100.0,
            "bootstrap_count": 4,
            "rescale_count": 10,
            "boundary_quality": 0.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {},
            "reserve_summary": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {
                "requested_budgets": 2,
                "candidate_solved_budgets": 1,
                "solved_budgets": 1,
                "requested_boundary_groups": 4,
                "solved_boundary_groups": 1,
                "candidate_solved_boundary_groups": 1,
                "unreachable_boundary_groups": 2,
                "fallback_selected_boundary_groups": 0,
            },
            "log_tail": "",
        }

    monkeypatch.setattr(oe_backend, "_policy_effect_summary", fake_policy_effect)
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)

    seed = evaluate_compile_candidate_program(context_path, seed_program)
    changed = evaluate_compile_candidate_program(context_path, changed_program)

    assert seed["metrics"]["placement_effect_score"] == 0.0
    assert changed["metrics"]["policy_effect_score"] == 1.0
    assert changed["metrics"]["placement_effect_score"] == 0.0
    assert changed["metrics"]["combined_score"] <= seed["metrics"]["combined_score"]


def test_sampled_effective_path_uses_sampled_seed_baseline():
    diagnostics = {
        "requested_boundary_groups": 1,
        "solved_boundary_groups": 1,
        "candidate_solved_boundary_groups": 1,
        "boundary_group_summaries": [
            {
                "group_key": {"pdag": "p0", "in_lvl": -1, "in_scl": 40},
                "requested_output_levels": [0, 1],
                "reachable_budgets": 2,
                "solved_budgets": 2,
                "candidate_solved_budgets": 2,
                "candidate_complete": True,
                "complete": True,
                "min_cost_usec": 100.0,
                "min_bootstrap": 1,
                "min_rescale": 2,
                "selected_source_counts": {"seed": 2},
            }
        ],
    }
    sampled_seed = {
        "valid": False,
        "bootstrap_count": 1,
        "rescale_count": 2,
        "sampled_selected_path_bootstraps": 1,
        "sampled_selected_path_rescales": 2,
        "sampled_selected_path_digest": "seed-sampled-digest",
        "diagnostics": diagnostics,
    }
    baseline = oe_backend._placement_baseline_from_result(sampled_seed)
    context = {
        "reference": {
            "effective_qbp_digest": "full-qbp-digest",
            "selected_path_digest": "full-path-digest",
        },
        "harness": {
            "eval_suite": "polybert-sampled",
            "sampled_seed_baseline": baseline,
        },
    }

    summary = oe_backend._result_effective_summary(context, sampled_seed, diagnostics)

    assert summary["selected_path_changed_vs_seed"] is False
    assert summary["effective_qbp_changed_vs_seed"] is False


def test_sampled_compile_eval_keeps_bounded_candidate_portfolio(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    hints = oe_backend._load_candidate_hints(program_path, context)

    sampled = oe_backend._compile_hints_for_eval_suite(hints, "polybert-sampled")
    full = oe_backend._compile_hints_for_eval_suite(hints, "polybert-full")

    assert sampled["strategy"] == hints["strategy"]
    assert len(sampled["portfolio"]) == 2
    assert sampled["allow_seed_fallback"] is True
    assert sampled["max_scale_candidates"] <= 16
    assert sampled["beam_width"] <= 4
    assert sampled["allow_bootstrap"] is False
    assert sampled["portfolio"][0]["strategy"] == "latency_beam"
    assert sampled["portfolio"][0]["budget_aggressive"] is True
    assert sampled["portfolio"][0]["allow_seed_fallback"] is True
    assert sampled["portfolio"][0]["state_cap_per_node"] <= 16
    assert sampled["portfolio"][0]["beam_width"] <= 6
    assert len(full["portfolio"]) == 9


def test_full_bundle_finalist_replay_bounds_portfolio(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    hints = oe_backend._load_candidate_hints(program_path, context)

    finalist = oe_backend._finalist_hints_for_full_bundle(hints)

    assert finalist["strategy"] == "level_preserving"
    assert "portfolio" not in finalist
    assert finalist["allow_seed_fallback"] is False
    assert finalist["max_scale_candidates"] <= 16


def test_bootstrap_mcts_full_replay_keeps_wide_boundary_caps(toy_cost_json: str):
    params = _params(
        toy_cost_json,
        openevolve_search_mode="bootstrap-mcts",
        openevolve_budget_aggressive=True,
        openevolve_target_bootstrap_count=9,
    )
    hints = oe_backend._bootstrap_mcts_seed_policy(params)
    hints.update(
        {
            "max_scale_candidates": 64,
            "boundary_state_cap": 8,
            "mcts_rollout_budget": 64,
            "mcts_action_cap": 16,
        }
    )

    finalist = oe_backend._finalist_hints_for_full_bundle(hints)

    assert finalist["strategy"] == "bootstrap_mcts"
    assert finalist["allow_seed_fallback"] is False
    assert finalist["max_scale_candidates"] == 64
    assert finalist["boundary_state_cap"] >= 6
    assert finalist["mcts_rollout_budget"] >= 32
    assert finalist["mcts_action_cap"] >= 12


def test_compile_fail_open_uses_validated_zero_iteration_portfolio(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    initial_hints = oe_backend._load_candidate_hints(program_path, context)

    fail_open = oe_backend._bounded_fail_open_hints(initial_hints)

    assert fail_open["fail_open_reason"] == "zero_iteration_seed_portfolio"
    assert fail_open["strategy"] == "level_preserving"
    assert isinstance(fail_open.get("portfolio"), list)
    assert {policy.get("min_internal_level") for policy in fail_open["portfolio"]} >= {6, 8, 12}


def test_target_bootstrap_score_rewards_absolute_progress_without_reference():
    high = oe_backend._target_bootstrap_score(9, 795, None)
    lower = oe_backend._target_bootstrap_score(9, 72, None)

    assert high > 0.0
    assert lower > high
    assert oe_backend._target_bootstrap_score(9, 9, None) == 1.0


def test_contextual_bootstrap_score_does_not_infer_component_targets(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    graph = Tdag(params, "unit_graph")
    graph.add_node("arg0", op="input", weight=1, op_descr={}, comment="")
    graph.add_node(
        "softmax",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.attention.self.qk_softmax.softmax_mul;op=qk_softmax_mul",
    )
    graph.add_edge("arg0", "softmax")
    graph.inputs = {"arg0"}
    graph.outputs = {"softmax"}
    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    lower = oe_backend._contextual_target_bootstrap_score(context, 9, 1, None)
    higher_but_under_explicit = oe_backend._contextual_target_bootstrap_score(context, 9, 3, None)

    assert lower == 1.0
    assert higher_but_under_explicit == 1.0
    assert context["unit_bootstrap_budget"]["unit_budgets"] == {}


def test_assignment_count_summary_does_not_sum_qbp_alternatives(monkeypatch):
    assignments = [object(), object(), object()]
    counts = iter(
        [
            {"bootstrap": 10, "rescale": 3},
            {"bootstrap": 20, "rescale": 5},
            {"bootstrap": 30, "rescale": 7},
        ]
    )
    monkeypatch.setattr(oe_backend, "_maintenance_counts", lambda _assign: next(counts))

    summary = oe_backend._assignment_count_summary(assignments)

    assert summary["aggregate_bootstrap"] == 60.0
    assert summary["avg_bootstrap"] == 20.0
    assert summary["min_bootstrap"] == 10.0
    assert summary["max_bootstrap"] == 30.0


def test_bootstrap_mcts_api_builds_low_bootstrap_actions(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)

    hints = PlacementMCTS(context).low_bootstrap_seed(target_bootstraps=9, action_cap=8)
    sampled = oe_backend._compile_hints_for_eval_suite(hints, "polybert-sampled")

    assert hints["strategy"] == "bootstrap_mcts"
    assert hints["target_bootstrap_count"] == 9
    assert len(hints["mcts_actions"]) >= 4
    assert hints["mcts_actions"][0]["name"] == "strict_no_bootstrap"
    assert any(
        action["name"] == "budget_fulfillment_beam"
        and action["policy"]["strategy"] == "latency_beam"
        and action["policy"]["direct_budget_policy"] is True
        for action in hints["mcts_actions"]
    )
    assert any(
        action["name"] == "waterline_budget_repair"
        and action["policy"]["strategy"] == "waterline_seed"
        and action["policy"]["direct_budget_policy"] is True
        for action in hints["mcts_actions"]
    )
    assert any(action["policy"].get("forbid_bootstrap") for action in hints["mcts_actions"])
    assert any(action["policy"]["allow_bootstrap"] for action in hints["mcts_actions"])
    assert sampled["mcts_rollout_budget"] <= 8
    assert sampled["mcts_action_cap"] <= 6
    assert sampled["mcts_max_repair_bootstraps"] <= 16
    capped_action_names = [
        action["name"]
        for action in sampled["mcts_actions"][: sampled["mcts_action_cap"]]
    ]
    assert "budget_fulfillment_beam" in capped_action_names
    sampled_latency_actions = [
        action
        for action in sampled["mcts_actions"]
        if action.get("policy", {}).get("strategy") == "latency_beam"
    ]
    sampled_latency_names = [action["name"] for action in sampled_latency_actions]
    assert sampled_latency_names[:2] == [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
    ]
    assert "profile_waterline_repair" in sampled_latency_names
    assert "tuneinsight_avgcase_cost_beam" in sampled_latency_names
    assert "tuneinsight_deferred_bootstrap_beam" in sampled_latency_names
    assert sampled_latency_actions[0]["policy"]["beam_width"] <= 5
    assert sampled_latency_actions[0]["policy"]["state_cap_per_node"] <= 16
    assert sampled_latency_actions[0]["policy"]["max_scale_candidates"] <= 32


def test_compile_evaluator_caps_bootstrap_mcts_sampled_rollouts(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_budget_aggressive=True,
        openevolve_target_bootstrap_count=9,
    )
    graph = _branch_merge_pdag(params)
    context_path = tmp_path / "compile_context.json"
    program_path = tmp_path / "candidate.py"
    context_path.write_text(json.dumps(build_compile_context(graph, params)), encoding="utf-8")
    program_path.write_text(
        "def place(context):\n"
        "    return {\n"
        "        'strategy': 'bootstrap_mcts',\n"
        "        'target_bootstrap_count': 9,\n"
        "        'mcts_rollout_budget': 48,\n"
        "        'mcts_action_cap': 32,\n"
        "        'budget_aggressive': True,\n"
        "    }\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_evaluate_compile_hints(_context, hints, suppress_output=False):
        captured.update(hints)
        assert suppress_output is True
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 10.0,
            "bootstrap_count": 3,
            "rescale_count": 2,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.1,
            "fallback_selected_budgets": 0,
            "boundary_quality": 1.0,
            "reserve_summary": {},
            "selected_output_state": {},
            "bottleneck_summary": [],
            "bootstrap_locations": [],
            "rescale_locations": [],
            "log_tail": "",
            "diagnostics": {
                "requested_budgets": 1,
                "candidate_solved_budgets": 1,
                "solved_budgets": 1,
            },
        }

    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)

    result = evaluate_compile_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert captured["strategy"] == "bootstrap_mcts"
    assert captured["mcts_rollout_budget"] <= 24
    assert captured["mcts_action_cap"] <= 12
    assert captured["mcts_max_repair_bootstraps"] <= 128
    assert captured["boundary_state_cap"] == 12


def test_bootstrap_mcts_rejects_rollouts_over_repair_cap(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    graph = _toy_pdag(params)

    class FakeAssign:
        def check_assign(self):
            return True

    monkeypatch.setattr(oe_backend, "build_conservative_assign", lambda *_args, **_kwargs: FakeAssign())
    monkeypatch.setattr(oe_backend, "_policy_assignment_score", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        oe_backend,
        "_aggregate_counts",
        lambda _assignments: {"bootstrap": 5, "rescale": 0},
    )

    with pytest.raises(oe_backend.PlacementError, match="bootstrap repair cap"):
        oe_backend._build_bootstrap_mcts_assign(
            graph,
            params,
            {"in_lvl": -1, "in_scl": 40},
            {
                "strategy": "bootstrap_mcts",
                "mcts_rollout_budget": 1,
                "mcts_action_cap": 1,
                "mcts_max_repair_bootstraps": 4,
            },
            None,
        )


def test_mcts_selection_expands_unvisited_actions_first():
    actions = [
        oe_backend.MCTSAction("visited", {"strategy": "level_preserving"}),
        oe_backend.MCTSAction("unvisited", {"strategy": "level_preserving"}),
    ]
    stats = [
        oe_backend._MCTSNodeStats(visits=3, reward_sum=3.0),
        oe_backend._MCTSNodeStats(),
    ]

    assert oe_backend._select_mcts_action(actions, stats, step=4, exploration_weight=1.4) == 1


def test_bootstrap_mcts_seed_solves_toy_without_more_bootstraps_than_legacy(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)
    budgets = [{"in_lvl": -1, "in_scl": 40}]
    legacy_diag = {}
    mcts_diag = {}

    solve_budget_batch(graph, budgets, le, params, oe_backend._zero_iteration_portfolio_hints(), legacy_diag)
    solve_budget_batch(graph, budgets, le, params, oe_backend._bootstrap_mcts_seed_policy(params), mcts_diag)

    legacy_counts = _aggregate_counts(legacy_diag["assignments"])
    mcts_counts = _aggregate_counts(mcts_diag["assignments"])
    assert mcts_diag["solved_budgets"] == 1
    assert mcts_counts["bootstrap"] <= legacy_counts["bootstrap"]


def test_bootstrap_mcts_batch_keeps_complete_boundary_group(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budgets = [
        {"in_lvl": -1, "in_scl": 40, "out_lvl": level}
        for level in range(1, params.lvl_ub + 1)
    ]
    captured_groups = []

    def fake_group_attempts(pdag, params_arg, group_budgets, le_arg, hints, diagnostics):
        captured_groups.append([budget["out_lvl"] for budget in group_budgets])
        attempts = {}
        for idx, budget in enumerate(group_budgets):
            attempt = oe_backend._solve_one_budget_attempt(
                pdag,
                params_arg,
                budget,
                le_arg,
                "candidate:boundary_mcts:test",
                oe_backend._default_policy_hints(),
            )
            attempts[idx] = [attempt]
        return attempts

    monkeypatch.setattr(oe_backend, "_boundary_mcts_group_attempts", fake_group_attempts)
    diagnostics = {}

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        budgets,
        le,
        params,
        {
            **oe_backend._bootstrap_mcts_seed_policy(params),
            "enable_direct_budget_beam": True,
            "allow_seed_fallback": False,
        },
        diagnostics,
    )

    assert captured_groups == [list(range(1, params.lvl_ub + 1))]
    assert diagnostics["requested_budgets"] == params.lvl_ub
    assert diagnostics["candidate_solved_budgets"] == params.lvl_ub
    assert diagnostics["fallback_selected_budgets"] == 0
    assert io_to_assign
    assert io_to_cost


def test_bootstrap_mcts_prefers_candidate_attempt_over_cheaper_fallback(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budget = {"in_lvl": -1, "in_scl": 40, "out_lvl": 1}

    def fake_group_attempts(pdag, params_arg, group_budgets, le_arg, hints, diagnostics):
        attempt = oe_backend._solve_one_budget_attempt(
            pdag,
            params_arg,
            group_budgets[0],
            le_arg,
            "candidate:boundary_mcts:expensive",
            oe_backend._default_policy_hints(),
        )
        return {
            0: [
                oe_backend._BudgetAttempt(
                    attempt.source,
                    attempt.assign,
                    attempt.cost + 1e30,
                    attempt.in_key,
                    attempt.out_key,
                    attempt.actual_cost,
                )
            ]
        }

    monkeypatch.setattr(oe_backend, "_boundary_mcts_group_attempts", fake_group_attempts)

    def fail_seed_fallback(_params):
        raise AssertionError("seed fallback should not run when candidate attempt exists")

    monkeypatch.setattr(oe_backend, "_seed_fallback_attempts", fail_seed_fallback)
    diagnostics = {}

    solve_budget_batch(
        graph,
        [budget],
        le,
        params,
        {**oe_backend._bootstrap_mcts_seed_policy(params), "allow_seed_fallback": True},
        diagnostics,
    )

    assert diagnostics["candidate_solved_budgets"] == 1
    assert diagnostics["fallback_selected_budgets"] == 0
    assert diagnostics["selected_source_counts"] == {"candidate:boundary_mcts:expensive": 1}


def test_bootstrap_mcts_final_compile_fail_opens_to_seed(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, openevolve_iterations=1)
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budgets = [{"in_lvl": -1, "in_scl": 40, "out_lvl": 1}]
    diagnostics = {}
    fallback_called = False

    def fake_boundary(*_args, **_kwargs):
        return {}, {}

    def fake_fast_seed(pdag, budget_list, le_arg, params_arg, diagnostics_arg):
        nonlocal fallback_called
        fallback_called = True
        return oe_backend.solve_budget_batch(
            pdag,
            budget_list,
            le_arg,
            params_arg,
            oe_backend._low_scale_frontier_policy(),
            diagnostics_arg,
        )

    monkeypatch.setattr(oe_backend, "_solve_budget_batch_boundary_mcts", fake_boundary)
    monkeypatch.setattr(oe_backend, "_solve_budget_batch_fast_seed", fake_fast_seed)

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        budgets,
        le,
        params,
        {
            **oe_backend._bootstrap_mcts_seed_policy(params),
            "enable_direct_budget_beam": True,
            "allow_seed_fallback": False,
        },
        diagnostics,
    )

    assert fallback_called
    assert diagnostics["fail_open_seed_replay"] is True
    assert io_to_assign
    assert io_to_cost


def test_bootstrap_mcts_can_opt_into_direct_budget_beam_before_actions(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    diagnostics = {}

    solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": params.Sw}],
        le,
        params,
        {
            **oe_backend._bootstrap_mcts_seed_policy(params),
            "enable_direct_budget_beam": True,
            "allow_seed_fallback": False,
        },
        diagnostics,
    )

    assert diagnostics["candidate_solved_budgets"] == 1
    assert diagnostics["fallback_selected_budgets"] == 0
    assert sum(diagnostics["selected_source_counts"].values()) == 1
    assert next(iter(diagnostics["selected_source_counts"])).startswith("candidate:")


def test_bootstrap_mcts_candidate_actions_precede_seed_repair(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    diagnostics = {}
    action_policy = oe_backend._budget_fulfillment_beam_policy()
    action_policy["direct_budget_policy"] = True

    solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": params.Sw}],
        le,
        params,
        {
            **oe_backend._bootstrap_mcts_seed_policy(params),
            "allow_seed_fallback": False,
            "include_seed_repair_actions": True,
            "mcts_actions": [
                {
                    "name": "budget_fulfillment_beam",
                    "policy": action_policy,
                    "prior": 1.0,
                }
            ],
        },
        diagnostics,
    )

    assert diagnostics["candidate_solved_budgets"] == 1
    assert diagnostics["fallback_selected_budgets"] == 0
    assert not any(
        source.startswith("seed_fallback")
        for source in diagnostics["selected_source_counts"]
    )


def test_bootstrap_mcts_skips_direct_budget_beam_by_default(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    diagnostics = {}

    solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": params.Sw}],
        le,
        params,
        oe_backend._bootstrap_mcts_seed_policy(params),
        diagnostics,
    )

    assert "candidate:direct_budget_beam" not in diagnostics["selected_source_counts"]


def test_sampled_seed_fallback_skips_latency_beam(toy_cost_json: str):
    params = _params(toy_cost_json)
    params.openevolve_evaluating_candidate = True
    params.openevolve_eval_suite = "polybert-sampled"

    sources = [source for source, _policy in oe_backend._seed_fallback_attempts(params)]

    assert "seed_fallback_latency_beam" not in sources
    assert sources == ["seed_fallback", "seed_fallback_relaxed", "seed_fallback_waterline"]


def test_compile_harness_retries_bypass_replay_without_bypass(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, bpsdepth=15)
    params.part = True
    params.openevolve_eval_suite = "polybert-full"
    graph = _toy_pdag(params)
    context = build_compile_context(graph, params)
    calls = []

    def fake_solve_partition(tdag, qbp_manager, prev_cost, le, params_arg):
        calls.append(params_arg.bpsdepth)
        if params_arg.bpsdepth is not None:
            raise AssertionError("Main PDAG #toy QBP not found in manager. Please add it first.")
        return oe_backend.solve_budget_batch(
            tdag,
            [{"in_lvl": -1, "in_scl": params_arg.Sw}],
            le,
            params_arg,
            oe_backend._low_scale_frontier_policy(),
            {},
        )

    monkeypatch.setattr(
        "scripts.optimizer.orbit.iterative_partition.solve_partition",
        fake_solve_partition,
    )

    result = oe_backend._evaluate_compile_hints(context, {}, suppress_output=True)

    assert result["valid"] is True
    assert calls == [15, None]
    assert "retrying with bypass disabled" in result["log_tail"]


def test_sampled_compile_harness_scores_partial_budget_progress(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json)
    params.openevolve_eval_suite = "polybert-sampled"
    graph = _toy_pdag(params)
    context = build_compile_context(graph, params)

    def fake_solve_partition(tdag, qbp_manager, prev_cost, le, params_arg):
        diagnostics = {}
        oe_backend.solve_budget_batch(
            tdag,
            [{"in_lvl": -1, "in_scl": params_arg.Sw}],
            le,
            params_arg,
            oe_backend._low_scale_frontier_policy(),
            diagnostics,
        )
        qbp_manager.openevolve_diagnostics.append(diagnostics)
        raise oe_backend.PlacementError("compile replay produced no valid final partitioning")

    monkeypatch.setattr(
        "scripts.optimizer.orbit.iterative_partition.solve_partition",
        fake_solve_partition,
    )

    result = oe_backend._evaluate_compile_hints(context, {}, suppress_output=True)

    assert result["valid"] is True
    assert result["sampled_progress_only"] is True
    assert result["validity"] == 1.0
    assert result["diagnostics"]["sampled_progress_only"] is True
    assert result["bootstrap_count"] >= 0


def test_sampled_compile_harness_uses_cached_budget_tasks(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json)
    params.openevolve_eval_suite = "polybert-sampled"
    graph = _toy_pdag(params)
    context = build_compile_context(graph, params)
    context["sampled_budget_tasks"] = [
        {
            "kind": "normal",
            "context": build_context(
                graph,
                [{"in_lvl": -1, "in_scl": params.Sw}],
                params,
            ),
        }
    ]

    def fake_solve_partition(*_args, **_kwargs):
        raise AssertionError("cached sampled eval must not replay solve_partition")

    monkeypatch.setattr(
        "scripts.optimizer.orbit.iterative_partition.solve_partition",
        fake_solve_partition,
    )

    result = oe_backend._evaluate_compile_hints(
        context,
        oe_backend._bootstrap_mcts_seed_policy(params),
        suppress_output=True,
    )

    assert result["valid"] is True
    assert result["sampled_progress_only"] is True
    assert result["diagnostics"]["sampled_direct_budget_eval"] is False
    assert result["diagnostics"]["sampled_qbp_group_eval"] is True
    assert result["diagnostics"]["sampled_task_count"] == 1
    assert result["diagnostics"]["solved_budgets"] == 1


def test_bootstrap_mcts_sampled_budget_sampler_preserves_boundary_groups(
    toy_cost_json: str,
):
    params = _params(
        toy_cost_json,
        openevolve_harness="compile",
        openevolve_search_mode="bootstrap-mcts",
        openevolve_eval_suite="polybert-sampled",
        openevolve_max_unit_samples=4,
    )
    params.openevolve_evaluating_candidate = True
    manager = QBPManager(params, LatencyEstimator(params))
    budgets = [
        {"in_lvl": -1, "in_scl": params.Sw, "out_lvl": out_lvl}
        for out_lvl in range(1, params.lvl_ub + 1)
    ]

    sampled = manager._sample_openevolve_eval_budgets(_toy_pdag(params), budgets)

    expected_levels = {1, max(1, params.bts_lb + 1), max(1, params.lvl_ub // 2), params.lvl_ub}
    assert len(sampled) == len(expected_levels)
    assert {int(item["out_lvl"]) for item in sampled} == expected_levels
    assert {
        (int(item["in_lvl"]), int(item["in_scl"]), str(item.get("maino_v", "")))
        for item in sampled
    } == {(-1, params.Sw, "")}


def test_compile_sampled_budget_cache_caps_groups_globally(toy_cost_json: str):
    params = _params(
        toy_cost_json,
        openevolve_harness="compile",
        openevolve_search_mode="bootstrap-mcts",
        openevolve_eval_suite="polybert-sampled",
        openevolve_max_unit_samples=5,
    )
    graph = _toy_pdag(params)

    class FakeManager:
        openevolve_budget_tasks = []

    for task_idx in range(3):
        budgets = []
        for group_idx in range(3):
            for out_lvl in (1, 4, 8, 16):
                budgets.append(
                    {
                        "in_lvl": task_idx * 10 + group_idx + 1,
                        "in_scl": params.Sw + group_idx,
                        "out_lvl": out_lvl,
                        "main_qbp_cost": {(out_lvl, params.Sw): float(task_idx * 100 + group_idx)},
                        "main_dag_size": task_idx + group_idx,
                    }
                )
        FakeManager.openevolve_budget_tasks.append(
            {"kind": "normal", "pdag": graph, "io_budgets": budgets}
        )

    tasks = oe_backend._sampled_budget_tasks_from_qbp_manager(FakeManager, params)
    sampled_groups = {}
    sampled_task_indexes = set()
    for task in tasks:
        sampled_task_indexes.add(task["index"])
        for budget in task["context"]["io_budgets"]:
            key = (
                task["index"],
                int(budget["in_lvl"]),
                int(budget["in_scl"]),
                str(budget.get("maino_v", "")),
            )
            sampled_groups.setdefault(key, set()).add(int(budget["out_lvl"]))

    assert 1 < len(sampled_task_indexes) <= 3
    assert len(sampled_groups) <= 5
    assert all(levels == {1, 4, 8, 16} for levels in sampled_groups.values())


def test_sampled_boundary_group_ranking_prefers_reachable_inputs(toy_cost_json: str):
    params = _params(toy_cost_json)
    reachable = (-1, params.Sw, "", 0)
    unreachable = (params.bts_lb - 1, params.Sw, "", 0)
    groups = {
        unreachable: [
            {
                "out_lvl": params.lvl_lb,
                "main_qbp_cost": {(params.lvl_lb, params.Sw): 1.0},
                "main_dag_size": 1,
            }
        ],
        reachable: [
            {
                "out_lvl": params.lvl_ub,
                "main_qbp_cost": {(params.lvl_ub, params.Sw): 10_000.0},
                "main_dag_size": 1,
            }
        ],
    }

    ranked = oe_backend._rank_sampled_boundary_group_keys(groups, params)

    assert ranked[0] == reachable
    assert ranked[-1] == unreachable


def test_bootstrap_mcts_returns_complete_qbp_boundary_group(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budgets = [
        {"in_lvl": -1, "in_scl": params.Sw, "out_lvl": out_lvl}
        for out_lvl in range(1, params.lvl_ub + 1)
    ]
    diagnostics = {}

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        budgets,
        le,
        params,
        {**oe_backend._bootstrap_mcts_seed_policy(params), "enable_direct_budget_beam": True},
        diagnostics,
    )

    assert io_to_assign
    assert io_to_cost
    out_levels = {
        out_lvl
        for out_to_cost in io_to_cost.values()
        for out_lvl, _out_scl in out_to_cost
    }
    assert out_levels == set(range(1, params.lvl_ub + 1))
    assert diagnostics["requested_boundary_groups"] == 1
    assert diagnostics["solved_boundary_groups"] == 1
    assert diagnostics["candidate_solved_boundary_groups"] == 1
    assert diagnostics["boundary_group_summaries"][0]["complete"] is True


def test_boundary_group_count_summary_uses_frontier_for_sampled_bootstraps():
    summary = oe_backend._boundary_group_count_summary(
        [
            {
                "solved_budgets": 4,
                "avg_bootstrap": 20.0,
                "avg_rescale": 12.0,
                "min_bootstrap": 3.0,
                "min_rescale": 4.0,
                "max_bootstrap": 32.0,
                "max_rescale": 20.0,
                "avg_cost_usec": 120.0,
                "min_cost_usec": 90.0,
                "max_cost_usec": 180.0,
            },
            {
                "solved_budgets": 2,
                "avg_bootstrap": 10.0,
                "avg_rescale": 6.0,
                "min_bootstrap": 1.0,
                "min_rescale": 2.0,
                "max_bootstrap": 18.0,
                "max_rescale": 9.0,
                "avg_cost_usec": 40.0,
                "min_cost_usec": 30.0,
                "max_cost_usec": 60.0,
            },
            {
                "solved_budgets": 0,
                "avg_bootstrap": 99.0,
                "avg_rescale": 99.0,
                "min_bootstrap": 99.0,
                "min_rescale": 99.0,
                "max_bootstrap": 99.0,
                "max_rescale": 99.0,
                "avg_cost_usec": 999.0,
                "min_cost_usec": 999.0,
                "max_cost_usec": 999.0,
            },
        ]
    )

    assert summary["boundary_group_count"] == 2.0
    assert summary["avg_bootstrap"] == 15.0
    assert summary["frontier_bootstrap"] == 2.0
    assert summary["frontier_rescale"] == 3.0
    assert summary["frontier_total_bootstrap"] == 4.0
    assert summary["frontier_total_rescale"] == 6.0
    assert summary["frontier_total_cost_usec"] == 120.0
    assert summary["avg_total_bootstrap"] == 30.0
    assert summary["avg_total_rescale"] == 18.0
    assert summary["avg_total_cost_usec"] == 160.0
    assert summary["min_cost_usec"] == 30.0
    assert summary["max_cost_usec"] == 180.0


def test_cost_minimization_objective_prefers_direct_lower_cost_candidate():
    context = {
        "harness": {
            "eval_suite": "polybert-sampled",
            "sampled_seed_baseline": {"objective_cost_usec": 1_000.0},
        },
        "reference": {"objective_cost_usec": 1_000.0},
    }
    diagnostics = {
        "requested_boundary_groups": 2,
        "solved_boundary_groups": 2,
        "candidate_solved_boundary_groups": 2,
        "fallback_selected_boundary_groups": 0,
        "invalid_boundary_groups": 0,
        "unreachable_boundary_groups": 0,
    }
    low_cost = {
        "valid": True,
        "final_latency_usec": 500.0,
        "objective_cost_usec": 500.0,
        "total_frontier_cost_usec": 500.0,
        "fallback_selected_budgets": 0,
    }
    high_cost = {**low_cost, "final_latency_usec": 2_000.0, "objective_cost_usec": 2_000.0}

    low_objective = oe_backend._cost_minimization_objective(
        context, low_cost, diagnostics, effective_validity=1.0, repair_count=0
    )
    high_objective = oe_backend._cost_minimization_objective(
        context, high_cost, diagnostics, effective_validity=1.0, repair_count=0
    )

    assert low_objective["objective_tier"] == 2.0
    assert low_objective["candidate_direct_group_coverage"] == 1.0
    assert low_objective["objective_cost_ratio_vs_seed"] == pytest.approx(2.0)
    assert low_objective["objective_cost_score"] > high_objective["objective_cost_score"]


def test_cost_minimization_objective_gates_partial_and_fallback_candidates():
    context = {
        "harness": {"seed_baseline": {"objective_cost_usec": 1_000.0}},
        "reference": {"objective_cost_usec": 1_000.0},
    }
    direct_result = {
        "valid": True,
        "final_latency_usec": 2_000.0,
        "total_frontier_cost_usec": 2_000.0,
        "fallback_selected_budgets": 0,
    }
    direct_diag = {
        "requested_boundary_groups": 2,
        "solved_boundary_groups": 2,
        "candidate_solved_boundary_groups": 2,
        "fallback_selected_boundary_groups": 0,
        "invalid_boundary_groups": 0,
        "unreachable_boundary_groups": 0,
    }
    fallback_diag = {**direct_diag, "fallback_selected_boundary_groups": 1}
    fallback_result = {**direct_result, "final_latency_usec": 250.0, "fallback_selected_budgets": 1}
    partial_diag = {
        **direct_diag,
        "solved_boundary_groups": 1,
        "candidate_solved_boundary_groups": 1,
    }
    partial_result = {**direct_result, "valid": False, "final_latency_usec": 100.0}

    direct = oe_backend._cost_minimization_objective(
        context, direct_result, direct_diag, effective_validity=1.0, repair_count=0
    )
    fallback = oe_backend._cost_minimization_objective(
        context, fallback_result, fallback_diag, effective_validity=1.0, repair_count=0
    )
    partial = oe_backend._cost_minimization_objective(
        context, partial_result, partial_diag, effective_validity=0.5, repair_count=0
    )

    assert direct["objective_tier"] == 2.0
    assert fallback["objective_tier"] == 1.0
    assert partial["objective_tier"] == 0.0
    assert fallback["objective_cost_usec"] > direct["objective_cost_usec"]
    assert partial["unsolved_reachable_boundary_groups"] == 1.0


def test_effective_path_digest_ignores_latency_but_tracks_qbp_path():
    diagnostics = {
        "requested_boundary_groups": 1,
        "solved_boundary_groups": 1,
        "candidate_solved_boundary_groups": 1,
        "boundary_group_summaries": [
            {
                "group_key": {"in_lvl": 16, "in_scl": 40, "maino_v": "", "main_dag_size": 0},
                "requested_output_levels": [1, 2],
                "reachable_budgets": 2,
                "solved_budgets": 2,
                "candidate_solved_budgets": 2,
                "fallback_selected_budgets": 0,
                "candidate_complete": True,
                "complete": True,
                "min_cost_usec": 10.0,
                "min_bootstrap": 1,
                "min_rescale": 2,
                "selected_source_counts": {"candidate:boundary_mcts:wide_boundary_cost_beam:0": 2},
            }
        ],
    }
    fast = {
        "valid": True,
        "final_latency_usec": 10.0,
        "objective_cost_usec": 10.0,
        "bootstrap_count": 1,
        "rescale_count": 2,
        "selected_output_state": {"out_lvl": 1, "out_scl": 40},
        "assignment": {"v_lvl_out": {"x": 1}},
    }
    slow = {**fast, "final_latency_usec": 99.0, "objective_cost_usec": 99.0}
    changed_diag = {
        **diagnostics,
        "boundary_group_summaries": [
            {
                **diagnostics["boundary_group_summaries"][0],
                "selected_source_counts": {"candidate:boundary_mcts:profile_waterline_repair:0": 2},
            }
        ],
    }

    assert oe_backend._selected_path_digest(fast, diagnostics) == oe_backend._selected_path_digest(slow, diagnostics)
    assert oe_backend._effective_qbp_digest(diagnostics) != oe_backend._effective_qbp_digest(changed_diag)


def test_sampled_path_proxy_exposes_selected_path_cost():
    summaries = [
        {
            "group_key": {"in_lvl": 16, "in_scl": 40, "maino_v": "", "main_dag_size": 0},
            "requested_output_levels": [1, 2],
            "solved_budgets": 2,
            "candidate_solved_budgets": 2,
            "min_cost_usec": 15.0,
            "min_bootstrap": 1,
            "min_rescale": 3,
            "selected_source_counts": {"candidate:direct_budget_beam": 2},
        },
        {
            "group_key": {"in_lvl": 15, "in_scl": 40, "maino_v": "", "main_dag_size": 0},
            "requested_output_levels": [1],
            "solved_budgets": 1,
            "candidate_solved_budgets": 1,
            "min_cost_usec": 7.0,
            "min_bootstrap": 0,
            "min_rescale": 1,
            "selected_source_counts": {"candidate:boundary_mcts:profile_waterline_repair:0": 1},
        },
    ]

    proxy = oe_backend._sampled_path_proxy_from_boundary_groups(summaries)

    assert proxy["sampled_dp_latency_usec"] == pytest.approx(22.0)
    assert proxy["sampled_selected_path_bootstraps"] == pytest.approx(1.0)
    assert proxy["sampled_selected_path_rescales"] == pytest.approx(4.0)
    assert proxy["sampled_selected_path_digest"]


def test_unrefreshable_empty_boundary_group_is_not_scored_invalid(toy_cost_json: str):
    params = _params(toy_cost_json)
    diagnostics = {"requested_boundary_groups": 0, "unreachable_boundary_groups": 0, "invalid_boundary_groups": 0}

    oe_backend._record_boundary_group_result(
        diagnostics,
        params,
        (1, params.Sf * 2, "", 0),
        [{"out_lvl": 1}],
        [],
        0,
        0,
    )

    assert diagnostics["requested_boundary_groups"] == 1
    assert diagnostics["unreachable_boundary_groups"] == 1
    assert diagnostics["invalid_boundary_groups"] == 0
    assert oe_backend._scored_boundary_group_count(diagnostics) == 1


def test_partial_group_with_unreachable_output_levels_counts_complete(
    toy_cost_json: str, monkeypatch
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    assign = Assign(graph)
    in_lvl = params.bts_lb + 1
    in_scl = params.Sf * in_lvl - (params.Sf * params.bts_lb - params.Sf) + 1
    group_key = (in_lvl, in_scl, "", 0)
    budgets = [
        {"out_lvl": params.lvl_lb},
        {"out_lvl": params.lvl_ub},
    ]
    attempt = oe_backend._BudgetAttempt(
        "candidate:unit-test",
        assign,
        10.0,
        (in_lvl, in_scl),
        (params.lvl_lb, params.Sw),
        10.0,
        {},
    )
    diagnostics = {
        "requested_boundary_groups": 0,
        "solved_boundary_groups": 0,
        "partial_boundary_groups": 0,
        "candidate_solved_boundary_groups": 0,
        "fallback_selected_boundary_groups": 0,
        "unreachable_boundary_groups": 0,
        "invalid_boundary_groups": 0,
        "boundary_group_summaries": [],
    }

    assert oe_backend._boundary_budget_output_cannot_refresh(params, group_key, budgets[1])
    monkeypatch.setattr(
        oe_backend,
        "_assignment_count_summary",
        lambda _assigns: {
            "avg_bootstrap": 0.0,
            "min_bootstrap": 0.0,
            "max_bootstrap": 0.0,
            "avg_rescale": 0.0,
            "min_rescale": 0.0,
            "max_rescale": 0.0,
        },
    )

    oe_backend._record_boundary_group_result(
        diagnostics,
        params,
        group_key,
        budgets,
        [attempt],
        1,
        0,
    )

    assert diagnostics["solved_boundary_groups"] == 1
    assert diagnostics["candidate_solved_boundary_groups"] == 1
    assert diagnostics["invalid_boundary_groups"] == 0
    summary = diagnostics["boundary_group_summaries"][0]
    assert summary["complete"] is True
    assert summary["reachable_budgets"] == 1
    assert summary["unreachable_budgets"] == 1


def test_boundary_scale_candidates_obey_output_level_bound(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    policy = oe_backend._with_default_policy(
        {
            "boundary_scale_policy": "sf",
            "max_scale_candidates": 32,
        }
    )

    candidates = oe_backend._boundary_scale_candidates(
        graph,
        params,
        {"in_lvl": -1, "in_scl": 40, "out_lvl": 1},
        policy,
    )

    assert candidates
    assert all(scale <= params.boundary_output_scale_bound(1) for scale in candidates)
    assert all(scale >= params.scale_lower_bound("0", graph.nodes["0"], "out") for scale in candidates)


def test_sampled_invalid_best_skips_expensive_full_bundle(tmp_path: Path):
    output_dir = tmp_path / "openevolve_output"
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True)
    (best_dir / "best_program_info.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "combined_score": 0.0925,
                    "validity": 0.0,
                    "effective_validity": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    assert oe_backend._sampled_best_invalid_reason(output_dir) == "sampled_best_solved_no_budgets"


def test_sampled_fallback_only_best_skips_expensive_full_bundle(tmp_path: Path):
    output_dir = tmp_path / "openevolve_output"
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True)
    (best_dir / "best_program_info.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "combined_score": 0.39,
                    "validity": 0.40,
                    "effective_validity": 0.40,
                    "candidate_qbp_coverage": 0.0,
                    "candidate_solved_boundary_groups": 0,
                    "fallback_selected_budgets": 34,
                    "fallback_selected_groups": 11,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        oe_backend._sampled_best_invalid_reason(output_dir)
        == "sampled_best_has_no_direct_qbp_coverage"
    )


def test_unreachable_boundary_groups_still_count_for_scoring():
    diagnostics = {
        "requested_boundary_groups": 16,
        "unreachable_boundary_groups": 6,
        "solved_boundary_groups": 10,
    }

    assert oe_backend._scored_boundary_group_count(diagnostics) == 16


def test_full_bundle_finalist_variants_include_bounded_portfolio(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    hints = oe_backend._load_candidate_hints(program_path, context)

    variants = oe_backend._finalist_hint_variants(hints)

    assert [label for label, _policy in variants][0] == "primary"
    assert len(variants) == 6
    assert any(label.startswith("portfolio_") for label, _policy in variants[1:])
    assert all("portfolio" not in policy for _label, policy in variants)


def test_layer_nonlinear_units_are_extracted_from_comments(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "unit_graph")
    graph.add_node("arg0", op="input", weight=1, op_descr={}, comment="")
    graph.add_node(
        "softmax",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.attention.self.qk_softmax.softmax_mul;op=qk_softmax_mul",
    )
    graph.add_node(
        "act",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.intermediate.gelu;op=quadratic_activation_square",
    )
    graph.add_edge("arg0", "softmax")
    graph.add_edge("softmax", "act")
    graph.inputs = {"arg0"}
    graph.outputs = {"act"}

    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    unit_ids = {unit["id"] for unit in context["placement_units"]}
    assert "layer:layer.0" in unit_ids
    assert "nonlinear:layer.0:attention_softmax" in unit_ids
    assert "nonlinear:layer.0:activation" in unit_ids
    assert context["unit_op_histogram"]["mul"] == 4
    budget = context["unit_bootstrap_budget"]
    pressure = budget["unit_maintenance_pressure"]["nonlinear:layer.0:attention_softmax"]
    assert pressure["pressure"] > 0
    assert budget["effective_target_bootstrap_count"] == 0


def test_layer_units_handle_prefixed_rotom_comments_and_waterline_summary(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "prefixed_rotom_comments")
    graph.add_node(
        "arg0",
        op="input",
        weight=1,
        op_descr={},
        comment="0 scope=fhe_bert.input;op=full_model_input",
    )
    graph.add_node(
        "softmax",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment=(
            "7716 scope=fhe_bert.bert.encoder.layer.1.attention.self."
            "qk_softmax.softmax_mul;op=qk_softmax_mul"
        ),
    )
    graph.add_node(
        "pool",
        op="add",
        weight=1,
        op_descr={"single": 1},
        comment="9910 scope=fhe_bert.bert.pooler.dense;op=pooler_dense",
    )
    graph.add_edge("arg0", "softmax")
    graph.add_edge("softmax", "pool")
    graph.inputs = {"arg0"}
    graph.outputs = {"pool"}

    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    unit_ids = {unit["id"] for unit in context["placement_units"]}
    assert "layer:layer.1" in unit_ids
    assert "nonlinear:layer.1:attention_softmax" in unit_ids
    assert "layer:pooler" in unit_ids
    waterline = context["waterline_profile"]
    assert waterline["ckks"]["input_waterline"] == 40
    assert waterline["whole_graph"]["output_scale_lower_bounds"] == {"40": 3}
    assert waterline["by_layer"]["layer.1"]["output_scale_lower_bounds"] == {"40": 1}
    assert waterline["by_layer"]["pooler"]["output_scale_lower_bounds"] == {"40": 1}


def test_component_bootstrap_alignment_is_neutral_without_count_budget(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "unit_graph")
    graph.add_node("arg0", op="input", weight=1, op_descr={}, comment="")
    graph.add_node(
        "softmax",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.attention.self.qk_softmax.softmax_mul;op=qk_softmax_mul",
    )
    graph.add_edge("arg0", "softmax")
    graph.inputs = {"arg0"}
    graph.outputs = {"softmax"}
    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    aligned = oe_backend._component_bootstrap_alignment_score(
        context,
        {"layer=layer.0;op=qk_softmax_mul": 3},
        3,
    )
    misplaced = oe_backend._component_bootstrap_alignment_score(
        context,
        {"layer=layer.0;op=plain_mul": 3},
        3,
    )

    assert aligned == misplaced
    assert aligned == 1.0


def test_mcts_candidate_actions_include_component_budget_repair(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "unit_graph")
    graph.add_node("arg0", op="input", weight=1, op_descr={}, comment="")
    graph.add_node(
        "softmax",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.attention.self.qk_softmax.softmax_mul;op=qk_softmax_mul",
    )
    graph.add_edge("arg0", "softmax")
    graph.inputs = {"arg0"}
    graph.outputs = {"softmax"}
    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    actions = candidate_actions(context, action_cap=8)
    component = next(action for action in actions if action["name"] == "component_budget_repair")

    assert component["policy"]["component_bootstrap_budgets"] == {}
    assert component["policy"]["bootstrap_anchor_count"] >= 1
    assert component["policy"]["selection_objective"] == "cost"
    assert component["policy"]["prefer_component_budget_fit"] is False
    assert component["policy"]["force_bootstrap_anchors"] is False


def test_nonlinear_phase_boundary_anchor_selector_prefers_phase_nodes(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "phase_anchor_graph")
    graph.add_node("arg0", op="input", weight=1, op_descr={}, comment="")
    phase_nodes = {
        "numerator_sum": "scope=fhe_bert.bert.encoder.layer.0.attention.self.qk_softmax;op=qk_numerator_sum",
        "row_sum": "scope=fhe_bert.bert.encoder.layer.0.attention.self.qk_softmax;op=qk_row_sum_add",
        "inv_seed": "scope=fhe_bert.bert.encoder.layer.0.attention.output.LayerNorm;op=inv_sqrt_seed",
        "newton_update": "scope=fhe_bert.bert.encoder.layer.0.attention.output.LayerNorm;op=inv_sqrt_newton_update",
    }
    excluded_nodes = {
        "denom_rescale": "scope=fhe_bert.bert.encoder.layer.0.attention.self.qk_softmax;op=denominator_rescale",
        "newton_xy": "scope=fhe_bert.bert.encoder.layer.0.attention.output.LayerNorm;op=inv_sqrt_newton_xy",
        "softmax_mul": "scope=fhe_bert.bert.encoder.layer.0.attention.self.qk_softmax;op=softmax_mul",
    }
    prev = "arg0"
    for node, comment in {**excluded_nodes, **phase_nodes}.items():
        graph.add_node(
            node,
            op="mul",
            weight=1,
            op_descr={"single": 1, "double": 0},
            comment=comment,
        )
        graph.add_edge(prev, node, weight=1)
        prev = node
    graph.inputs = {"arg0"}
    graph.outputs = {prev}

    anchors = oe_backend._bootstrap_anchor_nodes(
        graph,
        params,
        {},
        {
            "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
            "bootstrap_anchor_count": 8,
        },
    )

    assert set(phase_nodes).issubset(set(anchors))
    assert not set(excluded_nodes).intersection(anchors)


def test_candidate_actions_include_nonlinear_phase_boundary_policy(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": params.lvl_ub, "in_scl": params.Sw}], params)

    actions = candidate_actions(context, action_cap=12)
    phase = next(action for action in actions if action["name"] == "nonlinear_phase_boundary_beam")
    waterline = next(action for action in actions if action["name"] == "waterline_cost_beam")
    sampled = oe_backend._sampled_lightweight_mcts_actions(actions)
    sampled_names = [action["name"] for action in sampled]

    assert waterline["policy"]["boundary_scale_policy"] == "waterline"
    assert waterline["policy"]["selection_objective"] == "cost"
    assert "waterline_cost_beam" in sampled_names[:4]
    assert phase["policy"]["bootstrap_anchor_selector"] == "nonlinear_phase_boundaries"
    assert phase["policy"]["selection_objective"] == "cost"
    assert "nonlinear_phase_boundary_beam" in sampled_names[:8]


def test_sampled_boundary_group_limit_scales_with_parallelism(toy_cost_json: str):
    local_params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_max_unit_samples=64,
        openevolve_parallel_evaluations=1,
    )
    neptune_params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_max_unit_samples=64,
        openevolve_parallel_evaluations=8,
    )
    tiny_params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_max_unit_samples=10,
        openevolve_parallel_evaluations=8,
    )

    assert oe_backend._sampled_compile_boundary_group_limit(local_params) == 16
    assert oe_backend._sampled_compile_boundary_group_limit(neptune_params) == 32
    assert oe_backend._sampled_compile_boundary_group_limit(tiny_params) == 10


def test_compile_context_carries_evaluator_timeout(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_evaluator_timeout_sec=321)
    context = build_context(_toy_pdag(params), [{"in_lvl": params.lvl_ub, "in_scl": params.Sw}], params)

    assert context["params"]["openevolve_evaluator_timeout_sec"] == 321
    source = oe_backend._compile_evaluator_source(Path("/tmp/context.json"))
    assert "ctx.Process(" in source
    assert "_os.setsid()" in source
    assert "_os.killpg(proc.pid, _signal.SIGTERM)" in source
    assert "_os.killpg(proc.pid, _signal.SIGKILL)" in source
    assert "compile_evaluator_timeout" in source


def test_bootstrap_mcts_context_collection_uses_fast_seed(toy_cost_json: str):
    params = _params(
        toy_cost_json,
        openevolve_search_mode="bootstrap-mcts",
        openevolve_eval_suite="polybert-sampled",
    )
    initial = {
        "strategy": "bootstrap_mcts",
        "mcts_action_cap": 12,
        "mcts_rollout_budget": 24,
    }

    hints = oe_backend._context_collection_seed_hints(initial, params)

    assert hints is not initial
    assert hints["context_collection_seed"] is True
    assert hints["strategy"] == "bootstrap_mcts"
    assert hints["policy_bank_lightweight"] is True
    assert hints["budget_aggressive"] is True
    assert hints["mcts_rollout_budget"] == 4
    assert hints["mcts_action_cap"] == 2
    assert hints["mcts_action_allowlist"] == [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
    ]


def test_reference_json_adds_reference_boundary_action(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "reference_anchor_graph")
    graph.add_node("arg0", op="input", weight=1, op_descr={}, comment="")
    graph.add_node(
        "seed",
        op="mul",
        weight=1,
        op_descr={"single": 1, "double": 0},
        comment="scope=fhe_bert.bert.encoder.layer.0.attention.output.LayerNorm;op=inv_sqrt_seed",
    )
    graph.add_edge("arg0", "seed")
    graph.inputs = {"arg0"}
    graph.outputs = {"seed"}
    context = build_context(graph, [{"in_lvl": params.lvl_ub, "in_scl": params.Sw}], params)
    context.setdefault("harness", {})["reference_json"] = {
        "bootstrap_locations": {"layer=layer.0;op=inv_sqrt_seed": 2}
    }

    actions = candidate_actions(context, action_cap=12)
    reference = next(action for action in actions if action["name"] == "reference_boundary_cost_beam")
    anchors = oe_backend._bootstrap_anchor_nodes(graph, params, {}, reference["policy"])

    assert reference["policy"]["bootstrap_anchor_selector"] == "reference_bootstrap_locations"
    assert "inv_sqrt_seed" in reference["policy"]["bootstrap_anchor_include_patterns"]
    assert anchors == ["seed"]


def test_policy_bank_does_not_inject_model_specific_bootstrap_targets(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_search_mode="bootstrap-mcts")
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    initial = oe_backend._bootstrap_mcts_initial_policy_for_context(context)

    variants = oe_backend._compile_policy_bank_variants(initial, context, params)
    labels = [label for label, _hints in variants]
    waterline_hints = dict(variants[labels.index("waterline_direct_cost_compact")][1])
    waterline_presets = waterline_hints["mcts_action_presets"]

    def target_values(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "target_bootstrap_count":
                    yield item
                yield from target_values(item)
        elif isinstance(value, list):
            for item in value:
                yield from target_values(item)

    assert variants
    assert labels[0] == "waterline_direct_cost_compact"
    assert waterline_hints["selection_objective"] == "cost"
    assert waterline_hints["boundary_scale_policy"] == "waterline"
    assert waterline_hints["max_scale_candidates"] == 32
    assert waterline_hints["mcts_action_cap"] == 10
    assert [action["name"] for action in waterline_hints["mcts_actions"]] == [
        "strict_no_bootstrap",
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "profile_waterline_repair",
        "tuneinsight_avgcase_cost_beam",
        "tuneinsight_deferred_bootstrap_beam",
        "waterline_budget_repair",
        "component_budget_repair",
        "waterline_cost_beam",
        "dense_boundary_cost_beam",
    ]
    assert "waterline_cost_beam" not in waterline_hints["mcts_action_allowlist"]
    assert "waterline_cost_beam" not in waterline_presets
    assert set(waterline_presets) == {
        "budget_fulfillment_beam",
        "dense_boundary_cost_beam",
        "nonlinear_phase_boundary_beam",
        "wide_boundary_cost_beam",
    }
    assert waterline_presets["budget_fulfillment_beam"]["policy"]["bootstrap_penalty"] == 25_000_000.0
    assert waterline_presets["wide_boundary_cost_beam"]["policy"]["boundary_scale_policy"] == "waterline"
    assert all("gurobi" not in label.lower() for label, _ in variants)
    assert all("boundary40" not in label.lower() for label, _ in variants)
    assert all(
        int(raw or 0) == 0
        for _label, hints in variants
        for raw in target_values(hints)
    )


def test_policy_bank_includes_relaxed_scale_floor_variants(toy_cost_json: str):
    params = _params(
        toy_cost_json,
        openevolve_search_mode="bootstrap-mcts",
        scale_floor_policy="estimator-relaxed",
        openevolve_scale_floor_candidates="40,36,32,28",
    )
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    initial = oe_backend._bootstrap_mcts_initial_policy_for_context(context)

    variants = oe_backend._compile_policy_bank_variants(initial, context, params)
    labels = [label for label, _hints in variants]

    assert "relaxed_floor_36_frontier" in labels
    assert "relaxed_floor_32_frontier" in labels
    assert "relaxed_floor_36_waterline_boundary" in labels
    assert len(variants) > 12


def test_policy_bank_worker_count_uses_parallel_evaluations(
    toy_cost_json: str, monkeypatch: pytest.MonkeyPatch
):
    params = _params(
        toy_cost_json,
        openevolve_search_mode="bootstrap-mcts",
        openevolve_parallel_evaluations=8,
    )

    monkeypatch.delenv("ORBIT_OPENEVOLVE_POLICY_BANK_WORKERS", raising=False)
    assert oe_backend._policy_bank_worker_count(params, 22) == 4
    assert oe_backend._policy_bank_worker_count(params, 3) == 3

    monkeypatch.setenv("ORBIT_OPENEVOLVE_POLICY_BANK_WORKERS", "8")
    assert oe_backend._policy_bank_worker_count(params, 22) == 8

    monkeypatch.setenv("ORBIT_OPENEVOLVE_POLICY_BANK_WORKERS", "bad")
    assert oe_backend._policy_bank_worker_count(params, 22) == 4


def test_policy_bank_prepass_enabled_by_default(
    toy_cost_json: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    params = _params(
        toy_cost_json,
        openevolve_iterations=1,
        openevolve_search_mode="bootstrap-mcts",
    )
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    context_path = tmp_path / "compile_context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    initial = oe_backend._bootstrap_mcts_initial_policy_for_context(context)
    candidate = {**initial, "boundary_scale_policy": "waterline", "selection_objective": "cost"}

    monkeypatch.delenv("ORBIT_OPENEVOLVE_POLICY_BANK", raising=False)
    monkeypatch.setattr(
        oe_backend,
        "_compile_policy_bank_variants",
        lambda _initial, _context, _params: [("fast_waterline", candidate)],
    )
    evaluated_programs: list[Path] = []

    def fake_evaluate(_context_path, program_path):
        evaluated_programs.append(Path(program_path))
        assert "policy_bank_lightweight" not in Path(program_path).read_text(encoding="utf-8")
        return {
            "metrics": {
                "latency_only_correct": 1.0,
                "combined_score": 2.0,
                "objective_cost_usec": 90.0,
                "reference_objective_cost_usec": 100.0,
                "objective_improved_vs_seed": 1.0,
                "candidate_qbp_coverage": 1.0,
                "boundary_group_validity": 1.0,
                "fallback_selected_budgets": 0.0,
                "sampled_dp_latency_usec": 90.0,
                "sampled_selected_path_bootstraps": 2.0,
            },
            "artifacts": {
                "correctness_gate": {"reasons": []},
                "sampled_selected_path_digest": "policy-bank-path",
                "selected_path_digest": "policy-bank-path",
                "effective_qbp_digest": "policy-bank-qbp",
            },
        }

    monkeypatch.setattr(oe_backend, "evaluate_compile_candidate_program", fake_evaluate)

    result = oe_backend._run_compile_policy_bank_prepass(
        context_path,
        context,
        initial,
        tmp_path / "openevolve_output",
        params,
    )

    assert result["summary"]["enabled"] is True
    assert result["summary"]["selected_initial_label"] == "fast_waterline"
    assert result["initial_hints"]["policy_bank_validated_initial"] is True
    assert result["initial_hints"]["policy_bank_selected_label"] == "fast_waterline"
    assert result["initial_hints"]["boundary_scale_policy"] == "waterline"
    assert result["selected_record"]["objective_cost_usec"] == 90.0
    active = oe_backend._policy_bank_active_seed_baseline(result["selected_record"])
    assert active["objective_cost_usec"] == 90.0
    assert active["selected_path_digest"] == "policy-bank-path"
    assert evaluated_programs


def test_policy_bank_active_seed_becomes_latency_reference():
    active_result = {
        "sampled_selected_path_bootstraps": 2.0,
        "sampled_selected_path_rescales": 3.0,
    }
    active_digest = oe_backend._selected_path_digest(active_result, {})
    context = {
        "reference": {
            "objective_cost_usec": 100.0,
            "selected_path_digest": "old-path",
        },
        "harness": {
            "eval_suite": "polybert-sampled",
            "sampled_seed_baseline": {
                "objective_cost_usec": 100.0,
                "selected_path_digest": "old-path",
            },
            "active_seed_baseline": {
                "objective_cost_usec": 90.0,
                "selected_path_digest": active_digest[:24],
            },
        },
    }
    result = {
        "valid": True,
        "sampled_progress_only": True,
        "sampled_dp_latency_usec": 90.0,
    }
    diagnostics = {
        "requested_boundary_groups": 1,
        "solved_boundary_groups": 1,
        "candidate_solved_boundary_groups": 1,
    }

    objective = oe_backend._cost_minimization_objective(
        context,
        result,
        diagnostics,
        effective_validity=1.0,
        repair_count=0,
    )
    effective = oe_backend._result_effective_summary(
        context,
        active_result,
        {},
    )

    assert objective["reference_objective_cost_usec"] == 90.0
    assert oe_backend._objective_improved_vs_seed(objective) is False
    assert effective["seed_equivalent_path"] is True


def test_boundary_group_policy_overlay_targets_sampled_group():
    context = {
        "io_budgets": [
            {
                "in_lvl": 14,
                "in_scl": 40,
                "out_lvl": 12,
                "out_scl": 40,
                "main_dag_size": 0,
            }
        ]
    }
    group_key = (14, 40, "", 0)
    raw = [
        {
            "selector": {"in_lvl": 14, "in_scl": 40, "main_dag_size": 0},
            "policy": {
                "boundary_scale_policy": "waterline",
                "boundary_state_cap": 9,
                "bootstrap_penalty": 45_000_000.0,
            },
        }
    ]

    normalized = oe_backend._normalize_boundary_group_policies(raw, context)
    merged = oe_backend._apply_boundary_group_policy_overlays(
        {
            "boundary_scale_policy": "frontier",
            "mcts_action_allowlist": ["budget_fulfillment_beam"],
            "mcts_action_presets": {
                "budget_fulfillment_beam": {
                    "policy": {"boundary_scale_policy": "frontier", "boundary_state_cap": 4}
                }
            },
            "boundary_group_policies": normalized,
        },
        group_key,
    )

    assert len(normalized) == 1
    assert merged["boundary_scale_policy"] == "waterline"
    assert merged["boundary_state_cap"] == 9
    assert merged["bootstrap_penalty"] == 45_000_000.0
    budget_preset = merged["mcts_action_presets"]["budget_fulfillment_beam"]["policy"]
    assert budget_preset["boundary_scale_policy"] == "waterline"
    assert budget_preset["boundary_state_cap"] == 9
    assert budget_preset["bootstrap_penalty"] == 45_000_000.0


def test_placement_mcts_boundary_group_policy_defaults_to_direct_latency_beam():
    mcts = PlacementMCTS({"harness": {}})
    patch = mcts.boundary_group_policy(
        {"group_key": {"in_lvl": 8, "in_scl": 51, "maino_v": "", "main_dag_size": 0}},
        boundary_state_cap=12,
        max_scale_candidates=80,
    )

    policy = patch["policy"]
    assert policy["strategy"] == "latency_beam"
    assert policy["allow_bootstrap"] is True
    assert policy["allow_seed_fallback"] is False
    assert policy["direct_budget_policy"] is True
    assert policy["refresh_fanout_at_level_floor"] is True
    assert policy["min_transition_reserve"] == 0
    assert policy["min_decryptability_reserve"] == 0
    assert policy["boundary_state_cap"] == 12
    assert policy["max_scale_candidates"] == 80


def test_initial_compile_seed_uses_trace_boundary_group_policies(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json, openevolve_search_mode="bootstrap-mcts")
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    context["sampled_budget_tasks"] = [
        {
            "context": {
                "io_budgets": [
                    {
                        "in_lvl": params.lvl_ub,
                        "in_scl": params.Sw,
                        "out_lvl": params.lvl_ub - 1,
                        "out_scl": params.Sw,
                        "main_dag_size": 0,
                    }
                ]
            }
        }
    ]
    context["harness"]["top_costly_boundary_groups"] = [
        {
            "group_key": {
                "in_lvl": params.lvl_ub,
                "in_scl": params.Sw,
                "maino_v": "",
                "main_dag_size": 0,
            },
            "min_cost_usec": 123.0,
        }
    ]
    program_path = tmp_path / "initial_trace.py"
    program_path.write_text(
        oe_backend._initial_compile_program_source("bootstrap-mcts"),
        encoding="utf-8",
    )

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["boundary_group_policies"]
    first = hints["boundary_group_policies"][0]
    assert first["selector"]["in_lvl"] == params.lvl_ub
    assert first["policy"]["selection_objective"] == "cost"
    assert first["policy"]["boundary_state_cap"] >= 8


def test_policy_bank_record_carries_boundary_trace_examples():
    evaluation = {
        "metrics": {
            "latency_only_correct": 1.0,
            "objective_cost_usec": 90.0,
            "reference_objective_cost_usec": 100.0,
            "scale_floor_bits": 36.0,
        },
        "artifacts": {
            "correctness_gate": {"reasons": []},
            "execution_trace": json.dumps(
                {
                    "top_costly_boundary_groups": [
                        {
                            "group_key": {"in_lvl": 14, "in_scl": 40, "maino_v": "", "main_dag_size": 0},
                            "min_cost_usec": 12.0,
                        }
                    ],
                    "unsolved_boundary_groups": {"reasons": {}},
                }
            ),
        },
    }

    record = oe_backend._policy_bank_record("candidate", {"strategy": "bootstrap_mcts"}, evaluation)
    examples = oe_backend._policy_bank_candidate_examples(
        {"reference": {"objective_cost_usec": 100.0}},
        [record],
    )

    assert record["scale_floor_bits"] == 36.0
    assert record["top_costly_boundary_groups"][0]["group_key"]["in_lvl"] == 14
    assert examples[1]["top_costly_boundary_groups"][0]["min_cost_usec"] == 12.0


def test_estimator_relaxed_candidate_actions_include_scale_floors(toy_cost_json: str):
    params = _params(
        toy_cost_json,
        scale_floor_policy="estimator-relaxed",
        openevolve_scale_floor_candidates="40,36,32,28",
    )
    graph = _toy_pdag(params)
    context = build_context(graph, [{"in_lvl": params.lvl_ub, "in_scl": params.Sw}], params)

    actions = candidate_actions(context, action_cap=12)
    names = [action["name"] for action in actions]
    sampled_names = [
        action["name"] for action in oe_backend._sampled_lightweight_mcts_actions(actions)
    ]

    assert "estimator_relaxed_floor_36" in names
    assert "estimator_relaxed_floor_32" in names
    assert "estimator_relaxed_floor_28" in names
    assert "estimator_relaxed_floor_36" in sampled_names[:8]


def test_scale_floor_hints_normalize_to_configured_candidates(toy_cost_json: str):
    params = _params(
        toy_cost_json,
        scale_floor_policy="estimator-relaxed",
        openevolve_scale_floor_candidates="40,36,32,28",
    )

    assert oe_backend._scale_floor_bits_from_hints(params, {"scale_floor_bits": 34}) == 36
    assert oe_backend._scale_floor_bits_from_hints(params, {"scale_floor_bits": 27}) == 28
    assert oe_backend._scale_floor_bits_from_hints(params, {}) == 40


def test_assign_check_uses_active_estimator_relaxed_scale_floor(toy_cost_json: str):
    params = _params(
        toy_cost_json,
        scale_floor_policy="estimator-relaxed",
        openevolve_scale_floor_candidates="40,36,32,28",
    )
    graph = Tdag(params, "scale_floor_add")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    graph.add_node("0", op="add", weight=1, level=None, scale=None, op_descr={"single": 1}, comment="")
    graph.add_edge("arg0", "0", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"0"}
    assign = Assign(graph)
    for node in ("arg0", "0"):
        assign.v_lvl_out[node] = params.lvl_ub
        assign.v_scl_out[node] = 28
    assign.v_lvl_in["arg0"] = params.lvl_ub
    assign.v_scl_in["arg0"] = 28
    assign.e_lvl_out[("arg0", "0")] = params.lvl_ub
    assign.e_scl_out[("arg0", "0")] = 28

    with pytest.raises(ValueError, match="below local lower bound 40"):
        assign.check_assign()

    oe_backend._apply_active_scale_floor_from_hints(params, {"scale_floor_bits": 28})
    assert assign.check_assign() is True


def test_force_bootstrap_anchor_selects_refresh_transition(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "force_anchor")
    graph.add_node(
        "softmax",
        op="input",
        weight=1,
        op_descr={},
        comment="scope=bert.encoder.layer.0.attention.self.qk_softmax.softmax_mul;op=qk_softmax_mul",
    )
    graph.inputs = {"softmax"}
    graph.outputs = {"softmax"}
    policy = oe_backend._policy_options(
        {
            "strategy": "level_preserving",
            "allow_bootstrap": True,
            "force_bootstrap_nodes": ["softmax"],
            "bootstrap_penalty": 1_000_000_000.0,
        },
        params,
    )

    out_level, out_scale = oe_backend._choose_output_state(
        graph,
        "softmax",
        params,
        10,
        params.Sw,
        None,
        None,
        None,
        policy,
        LatencyEstimator(params),
    )

    assert not params.check_res(10, params.Sw, out_level, out_scale)


def test_component_budget_selection_prefers_budget_fit_over_low_cost(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=1)
    graph = Tdag(params, "selection_objective")
    graph.add_node(
        "softmax",
        op="input",
        weight=1,
        op_descr={},
        comment="scope=bert.encoder.layer.0.attention.self.qk_softmax.softmax_mul;op=qk_softmax_mul",
    )
    graph.inputs = {"softmax"}
    graph.outputs = {"softmax"}

    low = Assign(graph)
    low.v_lvl_in["softmax"] = params.lvl_ub
    low.v_scl_in["softmax"] = params.Sw
    low.v_lvl_out["softmax"] = params.lvl_ub
    low.v_scl_out["softmax"] = params.Sw

    maintained = Assign(graph)
    maintained.v_lvl_in["softmax"] = params.bts_lb
    maintained.v_scl_in["softmax"] = params.Sw
    maintained.v_lvl_out["softmax"] = params.bts_lb + 1
    maintained.v_scl_out["softmax"] = params.Sf

    policy = {
        "selection_objective": "component_budget_fit",
        "prefer_component_budget_fit": True,
        "target_bootstrap_count": 1,
        "component_bootstrap_budgets": {"nonlinear:layer.0:attention_softmax": 1},
    }
    low_attempt = oe_backend._BudgetAttempt(
        "candidate:low",
        low,
        1.0,
        (params.lvl_ub, params.Sw),
        (params.lvl_ub, params.Sw),
        1.0,
        policy,
    )
    maintained_attempt = oe_backend._BudgetAttempt(
        "candidate:maintained",
        maintained,
        10.0,
        (params.bts_lb, params.Sw),
        (params.bts_lb + 1, params.Sf),
        10.0,
        policy,
    )

    assert (
        oe_backend._best_candidate_attempt([low_attempt, maintained_attempt], params, policy)
        is maintained_attempt
    )


def test_min_bootstrap_selection_prefers_fewer_refreshes(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    graph = Tdag(params, "min_bootstrap_selection")
    graph.add_node("x", op="input", weight=1, op_descr={}, comment="scope=test.x")
    graph.inputs = {"x"}
    graph.outputs = {"x"}

    no_refresh = Assign(graph)
    no_refresh.v_lvl_in["x"] = params.lvl_ub
    no_refresh.v_scl_in["x"] = params.Sw
    no_refresh.v_lvl_out["x"] = params.lvl_ub
    no_refresh.v_scl_out["x"] = params.Sw

    refresh = Assign(graph)
    refresh.v_lvl_in["x"] = params.bts_lb
    refresh.v_scl_in["x"] = params.Sw
    refresh.v_lvl_out["x"] = params.bts_lb + 1
    refresh.v_scl_out["x"] = params.Sf

    policy = {"selection_objective": "min_bootstrap"}
    no_refresh_attempt = oe_backend._BudgetAttempt(
        "candidate:no_refresh",
        no_refresh,
        100.0,
        (params.lvl_ub, params.Sw),
        (params.lvl_ub, params.Sw),
        100.0,
        policy,
    )
    refresh_attempt = oe_backend._BudgetAttempt(
        "candidate:refresh",
        refresh,
        1.0,
        (params.bts_lb, params.Sw),
        (params.bts_lb + 1, params.Sf),
        1.0,
        policy,
    )

    assert (
        oe_backend._best_candidate_attempt([refresh_attempt, no_refresh_attempt], params, policy)
        is no_refresh_attempt
    )


def test_cost_root_ignores_exploratory_non_cost_action_objectives(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    graph = Tdag(params, "cost_root_selection")
    graph.add_node("x", op="input", weight=1, op_descr={}, comment="scope=test.x")
    graph.inputs = {"x"}
    graph.outputs = {"x"}

    low_cost = Assign(graph)
    low_cost.v_lvl_in["x"] = params.lvl_ub
    low_cost.v_scl_in["x"] = params.Sw
    low_cost.v_lvl_out["x"] = params.lvl_ub
    low_cost.v_scl_out["x"] = params.Sw

    non_cost = Assign(graph)
    non_cost.v_lvl_in["x"] = params.bts_lb
    non_cost.v_scl_in["x"] = params.Sw
    non_cost.v_lvl_out["x"] = params.bts_lb + 1
    non_cost.v_scl_out["x"] = params.Sf

    root_policy = {"selection_objective": "cost"}
    low_cost_attempt = oe_backend._BudgetAttempt(
        "candidate:cost",
        low_cost,
        1.0,
        (params.lvl_ub, params.Sw),
        (params.lvl_ub, params.Sw),
        1.0,
        {"selection_objective": "cost"},
    )
    non_cost_attempt = oe_backend._BudgetAttempt(
        "candidate:min_bootstrap",
        non_cost,
        10.0,
        (params.bts_lb, params.Sw),
        (params.bts_lb + 1, params.Sf),
        10.0,
        {"selection_objective": "min_bootstrap"},
    )

    assert (
        oe_backend._best_candidate_attempt(
            [non_cost_attempt, low_cost_attempt],
            params,
            root_policy,
        )
        is low_cost_attempt
    )


def test_component_budget_mcts_does_not_stop_at_too_few_bootstraps(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    hints = {
        "target_bootstrap_count": 9,
        "component_bootstrap_budgets": {
            "nonlinear:global:attention_softmax": 4,
            "nonlinear:global:norm": 2,
            "nonlinear:global:reciprocal": 2,
            "nonlinear:global:activation": 1,
        },
    }

    assert not oe_backend._policy_bootstrap_target_met(hints, params, 3)
    assert oe_backend._policy_bootstrap_target_met(hints, params, 7)


def test_context_includes_alphaevolve_guidance(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    graph = _toy_pdag(params)
    context = build_context(graph, [{"in_lvl": -1, "in_scl": params.Sw}], params)

    guidance = context["evolution_guidance"]

    assert guidance["source"].startswith("Adapting AlphaEvolve")
    assert "strategy_pool" in guidance
    assert any("execution-trace" in item or "trace" in item for item in guidance["principles"])
    assert guidance["requested_bootstrap_target"] == 9
    assert "maintenance_pressure_summary" in guidance


def test_alphaevolve_feedback_suggests_component_action_when_under_budget(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    graph = _toy_pdag(params)
    context = build_context(graph, [{"in_lvl": -1, "in_scl": params.Sw}], params)
    context["harness"] = {"target_bootstrap_count": 9}
    feedback = oe_backend._alphaevolve_feedback(
        context,
        {"mcts_action_cap": 8, "mcts_rollout_budget": 12},
        {
            "bootstrap_count": 3,
            "metrics": {
                "bootstrap_count": 3,
                "component_bootstrap_score": 0.5,
                "candidate_qbp_coverage": 1.0,
                "boundary_group_validity": 1.0,
                "fallback_selected_groups": 0,
            },
            "diagnostics": {
                "selected_source_counts": {"candidate:boundary_mcts:budget_fulfillment_beam:0": 8},
                "candidate_invalid_reasons": {
                    "component_budget_repair: PlacementError: no feasible incoming level/scale": 2
                },
            },
        },
    )

    joined = " ".join(feedback["suggestions"])
    assert "component pressure repair" in joined
    assert "force_bootstrap_anchors=False" in joined


def test_unit_policy_api_and_lenient_repairs(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "candidate.py"
    program_path.write_text(
        """
def place(context):
    return {
        "global_policy": {"strategy": "not-real", "max_scale_candidates": 100},
        "unit_policies": [
            {
                "selector": {"layer": "layer.0"},
                "policy": {"min_internal_level": 999, "beam_width": 99},
            },
            {"selector": {"unit_id": "missing"}, "policy": {"max_scale_candidates": 8}},
        ],
        "patches": [{"op": "prefer_node_scale", "node": "missing", "scale": 999}],
    }
""",
        encoding="utf-8",
    )

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["strategy"] == "level_preserving"
    assert hints["max_scale_candidates"] == 100
    assert len(hints["unit_policies"]) == 1
    assert hints["unit_policies"][0]["policy"]["min_internal_level"] == params.lvl_ub
    assert hints["unit_policies"][0]["policy"]["beam_width"] == 12
    assert hints["preferred_node_scales"] == {}
    assert oe_backend._repair_count(hints) >= 4


def test_unit_policy_shape_keeps_flat_policy_compatibility(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": -1, "in_scl": 40}], params)

    flat = oe_backend._normalize_candidate_hints({"max_scale_candidates": 7}, context)
    structured = oe_backend._normalize_candidate_hints(
        {
            "global_policy": {"max_scale_candidates": 9},
            "unit_policies": [
                {
                    "selector": {"layer": "layer.0"},
                    "policy": {"max_scale_candidates": 5},
                }
            ],
        },
        context,
    )

    assert flat["max_scale_candidates"] == 7
    assert structured["max_scale_candidates"] == 9
    assert structured["unit_policies"][0]["policy"]["max_scale_candidates"] == 5


def test_sparse_candidate_inherits_initial_mcts_presets(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": -1, "in_scl": 40}], params)
    context.setdefault("harness", {})["initial_policy_hints"] = {
        "strategy": "bootstrap_mcts",
        "boundary_scale_policy": "waterline",
        "mcts_action_presets": {
            "budget_fulfillment_beam": {
                "prior": 0.6,
                "policy": {
                    "beam_width": 12,
                    "boundary_scale_policy": "waterline",
                    "max_scale_candidates": 32,
                    "bootstrap_penalty": 25_000_000.0,
                    "selection_objective": "cost",
                },
            }
        },
    }

    hints = oe_backend._normalize_candidate_hints(
        {
            "mcts_action_presets": {
                "budget_fulfillment_beam": {
                    "policy": {"beam_width": 10},
                }
            }
        },
        context,
    )

    preset = hints["mcts_action_presets"]["budget_fulfillment_beam"]
    assert preset["prior"] == pytest.approx(0.6)
    assert preset["policy"]["beam_width"] == 10
    assert preset["policy"]["boundary_scale_policy"] == "waterline"
    assert preset["policy"]["max_scale_candidates"] == 32
    assert preset["policy"]["selection_objective"] == "cost"


def test_sampled_compile_eval_rejects_expensive_policy(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    context["harness"]["eval_suite"] = "polybert-sampled"

    assert oe_backend._sampled_policy_static_reasons(
        context,
        {"strategy": "waterline_seed"},
    ) == []

    context["harness"]["leniency"] = "strict"
    reasons = oe_backend._sampled_policy_static_reasons(
        context,
        {"strategy": "waterline_seed"},
    )

    assert "sampled policy rejects strategy 'waterline_seed'" in reasons


def test_discover_finalists_uses_best_and_checkpoint_scores(tmp_path: Path):
    output_dir = tmp_path / "openevolve_output"
    best_code = "def place(context):\n    return {'tag': 'result_best'}\n"
    high_code = "def place(context):\n    return {'tag': 'high_checkpoint'}\n"
    best_dir_code = "def place(context):\n    return {'tag': 'best_dir'}\n"
    low_code = "def place(context):\n    return {'tag': 'low_checkpoint'}\n"
    (output_dir / "best").mkdir(parents=True)
    (output_dir / "best" / "best_program.py").write_text(best_dir_code, encoding="utf-8")
    (output_dir / "best" / "best_program_info.json").write_text(
        json.dumps({"metrics": {"combined_score": 1.5}}),
        encoding="utf-8",
    )
    for name, code, score in (
        ("checkpoint_10", high_code, 1.9),
        ("checkpoint_20", low_code, 0.2),
    ):
        program_dir = output_dir / "checkpoints" / name / "programs"
        program_dir.mkdir(parents=True)
        (program_dir / "program.json").write_text(
            json.dumps({"code": code, "metrics": {"combined_score": score}}),
            encoding="utf-8",
        )

    finalists = oe_backend._discover_finalist_codes(output_dir, best_code, 3)

    assert finalists[0] == best_code
    assert high_code in finalists
    assert best_dir_code in finalists
    assert low_code not in finalists


def test_evaluator_validity_counts_duplicate_requested_budgets(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "candidate.py"
    budgets = [{"in_lvl": -1, "in_scl": 40}, {"in_lvl": -1, "in_scl": 40}]
    context_path.write_text(json.dumps(build_context(graph, budgets, params)), encoding="utf-8")
    program_path.write_text("def place(context):\n    return {}\n", encoding="utf-8")

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert result["metrics"]["solved_budgets"] == 2.0
    assert result["metrics"]["budget_count"] == 2.0


def test_evaluator_rejects_invalid_policy_score(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    good_program = tmp_path / "good.py"
    bad_program = tmp_path / "bad.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    good_program.write_text("def place(context):\n    return {}\n", encoding="utf-8")
    bad_program.write_text(
        "def place(context):\n"
        "    return {'preferred_node_scales': {'0': 9999}}\n",
        encoding="utf-8",
    )

    good = evaluate_candidate_program(context_path, good_program)
    bad = evaluate_candidate_program(context_path, bad_program)

    assert good["metrics"]["validity"] == 1.0
    assert bad["metrics"]["validity"] == 0.0
    assert bad["metrics"]["combined_score"] < good["metrics"]["combined_score"]


def test_partition_fallback_selected_candidate_scores_below_valid(
    toy_cost_json: str, tmp_path: Path, monkeypatch
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "fallback.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    program_path.write_text(
        "def place(context):\n    return {'strategy': 'level_preserving', 'allow_seed_fallback': True}\n",
        encoding="utf-8",
    )

    def fake_solve_budget_batch(_tdag, _budgets, _le, _params, _hints, diagnostics):
        diagnostics["solved_budgets"] = 1
        diagnostics["candidate_solved_budgets"] = 1
        diagnostics["fallback_selected_budgets"] = 1
        diagnostics["candidate_improved_budgets"] = 0
        diagnostics["costs"] = [10.0]
        diagnostics["candidate_costs"] = [30.0]
        diagnostics["assignments"] = []
        diagnostics["selected_source_counts"] = {"seed_fallback": 1}
        return {(-1, 40): {(16, 40): object()}}, {(-1, 40): {(16, 40): 10.0}}

    monkeypatch.setattr(oe_backend, "solve_budget_batch", fake_solve_budget_batch)
    monkeypatch.setattr(
        oe_backend,
        "_reference_metrics",
        lambda *_args, **_kwargs: {
            "avg_latency_usec": 10.0,
            "bootstrap_count": 1,
            "rescale_count": 1,
            "solved_budgets": 1,
        },
    )

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["fallback_selected_budgets"] == 1.0
    assert result["metrics"]["candidate_only_avg_latency_usec"] == 30.0
    assert json.loads(result["artifacts"]["selected_source_counts"]) == {"seed_fallback": 1}
    assert result["metrics"]["combined_score"] < 1.0
    assert result["metrics"]["combined_score"] > 0.0


def test_polybert_sampled_budgets_keep_bypass_and_cost_extremes(toy_cost_json: str):
    params = _params(toy_cost_json)
    params.openevolve_evaluating_candidate = True
    params.openevolve_eval_suite = "polybert-sampled"
    params.openevolve_search_mode = "legacy"
    manager = QBPManager(params, None)
    budgets = []
    for group_idx in range(20):
        for out_lvl in range(1, params.lvl_ub + 1):
            budget = {
                "in_lvl": group_idx % 4,
                "in_scl": 20 + group_idx,
                "out_lvl": out_lvl,
                "main_dag_size": group_idx,
                "main_qbp_cost": {(1, 20): float(group_idx)},
            }
            if group_idx in {0, 19}:
                budget["maino_v"] = f"fork_{group_idx}"
            budgets.append(budget)

    sampled = manager._sample_openevolve_eval_budgets(budgets)

    assert 0 < len(sampled) <= params.openevolve_max_unit_samples
    assert any("maino_v" in budget for budget in sampled)
    assert max(min(budget["main_qbp_cost"].values()) for budget in sampled) == 19.0
    by_group = defaultdict(set)
    for budget in sampled:
        by_group[(budget["in_lvl"], budget["in_scl"], budget.get("maino_v", ""))].add(
            budget["out_lvl"]
        )
    assert all(levels == set(range(1, params.lvl_ub + 1)) for levels in by_group.values())


def test_evaluator_uses_quality_as_partial_validity_tiebreaker(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=4)
    context_path = tmp_path / "context.json"
    seed_program = tmp_path / "seed.py"
    evolved_program = tmp_path / "evolved.py"
    budgets = [
        {"in_lvl": -1, "in_scl": 40},
        {"in_lvl": 1, "in_scl": 9999},
    ]
    context_path.write_text(json.dumps(build_context(graph, budgets, params)), encoding="utf-8")
    seed_program.write_text(
        "def place(context):\n    return {'strategy': 'waterline_seed'}\n",
        encoding="utf-8",
    )
    evolved_program.write_text(
        "def place(context):\n"
        "    return {'strategy': 'level_preserving', 'refresh_fanout_at_level_floor': True}\n",
        encoding="utf-8",
    )

    seed = evaluate_candidate_program(context_path, seed_program)
    evolved = evaluate_candidate_program(context_path, evolved_program)

    assert seed["metrics"]["validity"] == evolved["metrics"]["validity"] == 0.5
    assert evolved["metrics"]["avg_latency_usec"] < seed["metrics"]["avg_latency_usec"]
    assert evolved["metrics"]["combined_score"] > seed["metrics"]["combined_score"]
    assert evolved["metrics"]["combined_score"] < 1.0


def test_fail_open_rejects_sampled_only_policy_bank_seed():
    initial = {
        "strategy": "bootstrap_mcts",
        "policy_bank_validated_initial": True,
        "policy_bank_selected_label": "latency_beam_waterline48_compact",
        "policy_bank_lightweight": True,
        "boundary_state_cap": 8,
        "mcts_actions": [
            {"name": "budget_fulfillment_beam", "policy": {"strategy": "latency_beam"}},
            {"name": "dense_boundary_cost_beam", "policy": {"strategy": "latency_beam"}},
        ],
        "mcts_action_allowlist": [
            "budget_fulfillment_beam",
            "dense_boundary_cost_beam",
        ],
        "policy_bank_selected_top_costly_boundary_groups": [
            {
                "group_key": {
                    "in_lvl": 8,
                    "in_scl": 51,
                    "maino_v": "",
                    "main_dag_size": 0,
                },
                "selected_source_counts": {
                    "candidate:boundary_mcts:dense_boundary_cost_beam:0": 4,
                    "candidate:boundary_mcts:budget_fulfillment_beam:0": 1,
                },
            }
        ],
    }

    fallback = oe_backend._bounded_fail_open_hints(initial)

    assert fallback["fail_open_reason"] == "zero_iteration_seed_portfolio"
    assert fallback["rejected_initial_reason"] == "sampled_policy_bank_not_full_validated"
    assert fallback["rejected_policy_bank_selected_label"] == "latency_beam_waterline48_compact"
    assert fallback["strategy"] == "level_preserving"
    assert "policy_bank_full_compile_replay" not in fallback


def test_fail_open_preserves_full_validated_policy_bank_seed():
    initial = {
        "strategy": "bootstrap_mcts",
        "policy_bank_validated_initial": True,
        "policy_bank_full_validated_initial": True,
        "policy_bank_selected_label": "latency_beam_waterline48_compact",
        "policy_bank_lightweight": True,
        "boundary_state_cap": 8,
        "mcts_actions": [
            {"name": "budget_fulfillment_beam", "policy": {"strategy": "latency_beam"}},
            {"name": "dense_boundary_cost_beam", "policy": {"strategy": "latency_beam"}},
        ],
        "mcts_action_allowlist": [
            "budget_fulfillment_beam",
            "dense_boundary_cost_beam",
        ],
        "policy_bank_selected_top_costly_boundary_groups": [
            {
                "group_key": {
                    "in_lvl": 8,
                    "in_scl": 51,
                    "maino_v": "",
                    "main_dag_size": 0,
                },
                "selected_source_counts": {
                    "candidate:boundary_mcts:dense_boundary_cost_beam:0": 4,
                    "candidate:boundary_mcts:budget_fulfillment_beam:0": 1,
                },
            }
        ],
    }

    fallback = oe_backend._bounded_fail_open_hints(initial)

    assert fallback["fail_open_reason"] == "policy_bank_full_validated_seed"
    assert fallback["policy_bank_selected_label"] == "latency_beam_waterline48_compact"
    assert fallback["boundary_state_cap"] == 8
    assert fallback["policy_bank_full_compile_replay"] is True
    assert fallback["mcts_action_allowlist"] == ["budget_fulfillment_beam"]
    assert fallback["mcts_action_cap"] == 1
    assert fallback["mcts_rollout_budget"] == 1
    assert fallback["boundary_group_policies"] == [
        {
            "selector": {
                "in_lvl": 8,
                "in_scl": 51,
                "maino_v": "",
                "main_dag_size": 0,
            },
            "policy": {
                "mcts_action_allowlist": ["dense_boundary_cost_beam"],
                "mcts_action_cap": 1,
                "mcts_rollout_budget": 1,
            },
        }
    ]
    assert "policy_bank_lightweight" not in fallback


def test_policy_bank_replay_hints_extract_dominant_boundary_action():
    assert (
        oe_backend._dominant_boundary_mcts_action(
            {
                "candidate:boundary_mcts:wide_boundary_cost_beam:0": 2,
                "candidate:boundary_mcts:dense_boundary_cost_beam:0": 5,
                "seed_fallback_latency_beam": 99,
            }
        )
        == "dense_boundary_cost_beam"
    )
    assert oe_backend._dominant_boundary_mcts_action({"seed_fallback": 3}) is None


def test_parallel_qbp_worker_count_uses_threads_and_env(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json)
    params.threads = 96
    assert oe_backend._parallel_qbp_worker_count(params, 100) == 32
    assert oe_backend._parallel_qbp_worker_count(params, 7) == 7
    monkeypatch.setenv("ORBIT_OPENEVOLVE_QBP_WORKERS", "12")
    assert oe_backend._parallel_qbp_worker_count(params, 100) == 12
    monkeypatch.setenv("ORBIT_OPENEVOLVE_QBP_WORKERS", "bad")
    assert oe_backend._parallel_qbp_worker_count(params, 100) == 32


def test_compile_harness_scores_final_compile_latency(toy_cost_json: str, tmp_path: Path):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="toy",
    )
    graph = _branch_merge_pdag(params)
    context = build_compile_context(graph, params)
    context["harness"]["trace_dir"] = str(tmp_path / "trace_repository")
    context_path = tmp_path / "compile_context.json"
    seed_program = tmp_path / "seed.py"
    evolved_program = tmp_path / "evolved.py"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    seed_program.write_text("def place(context):\n    return {}\n", encoding="utf-8")
    evolved_program.write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).level_preserving()\n",
        encoding="utf-8",
    )

    seed = evaluate_compile_candidate_program(context_path, seed_program)
    evolved = evaluate_compile_candidate_program(context_path, evolved_program)

    assert seed["metrics"]["validity"] == 1.0
    assert evolved["metrics"]["validity"] == 1.0
    assert "candidate_final_latency_usec" in evolved["artifacts"]
    assert evolved["metrics"]["final_latency_usec"] > 0.0
    if evolved["metrics"].get("latency_only_correct") == 1.0:
        candidate_artifacts = json.loads(evolved["artifacts"]["candidate_mlir_artifacts"])
        assert candidate_artifacts["written"] is True
        assert Path(candidate_artifacts["mlir_path"]).exists()
        assert candidate_artifacts["mlir_digest"]
        assert "earth." in evolved["artifacts"]["candidate_mlir_preview"]


def test_latency_gate_keeps_seed_equivalent_path_as_correct_baseline():
    result = {
        "valid": True,
        "fallback_selected_budgets": 0,
    }
    diagnostics = {
        "fallback_selected_boundary_groups": 0,
        "invalid_boundary_groups": 0,
    }
    objective = {
        "direct_complete": True,
        "objective_cost_usec": 1000.0,
        "reference_objective_cost_usec": 1000.0,
        "base_objective_cost_usec": 1000.0,
    }
    gate = oe_backend._latency_only_correctness_gate(
        result,
        diagnostics,
        objective,
        static={"valid": True},
        boundary_group_validity=1.0,
        candidate_qbp_coverage=1.0,
        effective_path={
            "reference_selected_path_digest": "seed",
            "selected_path_digest": "seed",
            "selected_path_changed_vs_seed": False,
            "effective_qbp_changed_vs_seed": False,
            "seed_equivalent_path": True,
        },
        policy_effect={"seed_equivalent": False},
    )

    assert gate["correct"] is True
    assert gate["objective_improved_vs_seed"] is False
    assert gate["seed_equivalent_path"] is True
    assert gate["reasons"] == []


def test_latency_gate_allows_equivalent_path_with_real_latency_improvement():
    result = {
        "valid": True,
        "fallback_selected_budgets": 0,
    }
    diagnostics = {
        "fallback_selected_boundary_groups": 0,
        "invalid_boundary_groups": 0,
    }
    objective = {
        "direct_complete": True,
        "objective_cost_usec": 900.0,
        "reference_objective_cost_usec": 1000.0,
        "base_objective_cost_usec": 1000.0,
    }
    gate = oe_backend._latency_only_correctness_gate(
        result,
        diagnostics,
        objective,
        static={"valid": True},
        boundary_group_validity=1.0,
        candidate_qbp_coverage=1.0,
        effective_path={
            "reference_selected_path_digest": "seed",
            "selected_path_digest": "seed",
            "selected_path_changed_vs_seed": False,
            "effective_qbp_changed_vs_seed": False,
            "seed_equivalent_path": True,
        },
        policy_effect={"seed_equivalent": False},
    )

    assert gate["correct"] is True
    assert gate["objective_improved_vs_seed"] is True


def test_latency_only_score_tiers_slower_candidates_below_improvements():
    slower = {
        "objective_cost_usec": 1001.0,
        "reference_objective_cost_usec": 1000.0,
        "base_objective_cost_usec": 1000.0,
    }
    improved = {
        "objective_cost_usec": 999.0,
        "reference_objective_cost_usec": 1000.0,
        "base_objective_cost_usec": 1000.0,
    }
    equal = {
        "objective_cost_usec": 1000.0,
        "reference_objective_cost_usec": 1000.0,
        "base_objective_cost_usec": 1000.0,
    }
    much_better = {
        "objective_cost_usec": 900.0,
        "reference_objective_cost_usec": 1000.0,
        "base_objective_cost_usec": 1000.0,
    }

    slower_score = oe_backend._latency_only_combined_score(slower, correct=True)
    equal_score = oe_backend._latency_only_combined_score(equal, correct=True)
    improved_score = oe_backend._latency_only_combined_score(improved, correct=True)
    much_better_score = oe_backend._latency_only_combined_score(much_better, correct=True)

    assert slower_score == pytest.approx(1000.0 / 1001.0)
    assert equal_score == 1.0
    assert improved_score > 1.0
    assert 0.0 < slower_score < equal_score
    assert equal_score > slower_score
    assert improved_score > slower_score
    assert much_better_score > improved_score
    assert oe_backend._latency_only_combined_score(improved, correct=False) == 0.0


def test_sampled_best_without_latency_improvement_is_rejected(tmp_path: Path):
    best_dir = tmp_path / "best"
    best_dir.mkdir()
    (best_dir / "best_program_info.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "validity": 1.0,
                    "effective_validity": 1.0,
                    "combined_score": 0.999,
                    "latency_only_correct": 1.0,
                    "seed_equivalent_path": 0.0,
                    "objective_improved_vs_seed": 0.0,
                    "candidate_qbp_coverage": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    assert oe_backend._sampled_best_invalid_reason(tmp_path) == "sampled_best_not_latency_improved"


def test_compile_invalid_metrics_cover_configured_feature_dimensions(
    toy_cost_json: str,
    tmp_path: Path,
):
    params = _params(toy_cost_json, openevolve_eval_suite="toy")
    graph = _toy_pdag(params)
    context_path = tmp_path / "compile_context.json"
    bad_program = tmp_path / "bad.py"
    context_path.write_text(json.dumps(build_compile_context(graph, params)), encoding="utf-8")
    bad_program.write_text(
        "def place(context):\n"
        "    return {'preferred_node_scales': {'0': 9999}}\n",
        encoding="utf-8",
    )

    result = evaluate_compile_candidate_program(context_path, bad_program)

    for key in (
        "final_latency_usec",
        "boundary_quality",
        "bootstrap_count",
        "component_bootstrap_score",
        "candidate_qbp_coverage",
        "boundary_group_validity",
        "placement_effect_score",
        "policy_effect_score",
        "action_effect_score",
        "seed_equivalent_policy",
        "rescale_count",
        "fallback_selected_budgets",
        "profile_risk",
        "placement_runtime_sec",
        "estimated_precision_bits",
        "output_margin_bits",
    ):
        assert key in result["metrics"]


def test_generated_gemini_config_defaults(toy_cost_json: str, monkeypatch):
    _install_fake_openevolve(monkeypatch, lambda **_kwargs: None)
    params = _params(toy_cost_json, openevolve_iterations=50, openevolve_seed=7)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))

    config = worker._openevolve_config_arg()

    assert config.random_seed == 7
    assert config.max_code_length == 50_000
    assert config.diff_based_evolution is False
    assert config.database.random_seed == 7
    assert config.llm.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.llm.models[0].name == "gemini-3.1-flash-lite"
    assert config.llm.models[0].api_base == config.llm.api_base
    assert config.llm.models[0].random_seed == 7
    assert config.llm.timeout == 180
    assert config.llm.retries == 1
    assert config.llm.retry_delay == 2
    assert config.llm.max_tokens == 50_000
    assert config.llm.models[0].timeout == 180
    assert config.evaluator.timeout == 180
    assert config.evaluator.parallel_evaluations == 1
    assert config.checkpoint_interval == 5
    assert config.prompt.max_artifact_bytes == 48 * 1024
    assert config.database.feature_dimensions == [
        "objective_cost_usec",
        "objective_cost_ratio_vs_seed",
        "boundary_group_validity",
        "candidate_direct_group_coverage",
        "unsolved_reachable_boundary_groups",
        "total_frontier_cost_usec",
        "final_latency_usec",
        "bootstrap_count",
        "component_bootstrap_score",
        "placement_effect_score",
        "policy_effect_score",
        "action_effect_score",
        "seed_equivalent_policy",
        "rescale_count",
        "fallback_selected_budgets",
        "profile_risk",
        "placement_runtime_sec",
        "estimated_precision_bits",
        "output_margin_bits",
    ]


def test_generated_multi_provider_config_uses_per_model_keys(toy_cost_json: str, monkeypatch):
    _install_fake_openevolve(monkeypatch, lambda **_kwargs: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    params = _params(
        toy_cost_json,
        openevolve_iterations=100,
        openevolve_provider="gemini",
        openevolve_model="gemini-3.1-flash-lite",
        openevolve_primary_weight=0.7,
        openevolve_secondary_provider="openai",
        openevolve_secondary_model="gpt-5.5",
        openevolve_secondary_weight=0.3,
    )
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))

    config = worker._openevolve_config_arg()

    assert [model.name for model in config.llm.models] == [
        "gemini-3.1-flash-lite",
        "gpt-5.5",
    ]
    assert config.llm.models[0].api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.llm.models[1].api_base == "https://api.openai.com/v1"
    assert config.llm.models[0].api_key == "test-gemini-key"
    assert config.llm.models[1].api_key == "test-openai-key"
    assert config.llm.models[0].weight == pytest.approx(0.7)
    assert config.llm.models[1].weight == pytest.approx(0.3)


def test_explicit_openevolve_config_preserves_model_and_updates_seed(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    loaded = FakeConfig()
    loaded.llm.models = [FakeLLMModelConfig(name="custom-model", api_base="https://custom.invalid/v1")]
    loaded.llm.evaluator_models = [
        FakeLLMModelConfig(name="custom-evaluator", api_base="https://custom.invalid/v1")
    ]
    _install_fake_openevolve(monkeypatch, lambda **_kwargs: None, loaded_config=loaded)
    params = _params(
        toy_cost_json,
        openevolve_config=str(tmp_path / "config.yaml"),
        openevolve_model="gemini-3.1-pro-preview",
        openevolve_seed=99,
    )
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))

    config = worker._openevolve_config_arg()

    assert config is loaded
    assert config.random_seed == 99
    assert config.database.random_seed == 99
    assert config.llm.models[0].name == "custom-model"
    assert config.llm.evaluator_models[0].name == "custom-evaluator"
    assert config.llm.models[0].random_seed == 99


def test_openevolve_env_uses_openai_key(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=2)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with worker._openevolve_runtime_env():
        assert os.environ["OPENAI_API_KEY"] == "test-openai-key"


def test_openevolve_env_gemini_fallback_bridges_openai(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=2)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    with worker._openevolve_runtime_env():
        assert os.environ["OPENAI_API_KEY"] == "test-gemini-key"

    assert "OPENAI_API_KEY" not in os.environ


def test_openevolve_env_missing_key_has_clear_error(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=2)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError) as excinfo:
        with worker._openevolve_runtime_env():
            pass

    msg = str(excinfo.value)
    assert "OPENAI_API_KEY" in msg
    assert "GEMINI_API_KEY" in msg
    assert "AIza" not in msg


def test_mocked_openevolve_runtime_runs_once_per_budget_batch(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    class FakeResult:
        best_code = (
            "def place(context):\n"
            "    return {'strategy': 'waterline_seed'}\n"
        )

    def fake_run_evolution(**kwargs):
        assert os.environ["OPENAI_API_KEY"] == "test-gemini-key"
        calls.append(kwargs)
        return FakeResult()

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gurobipy" or name.startswith("gurobipy.") or name == "pulp":
            raise AssertionError("OpenEvolve placement imported an ILP solver")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    params = _params(
        toy_cost_json,
        openevolve_iterations=3,
        openevolve_output_dir=str(tmp_path),
    )
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    default_assign, _default_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}, {"in_lvl": -1, "in_scl": 41}],
        LatencyEstimator(params),
        params,
    )

    io_to_assign, io_to_cost = worker.get_qbp(
        graph,
        [{"in_lvl": -1, "in_scl": 40}, {"in_lvl": -1, "in_scl": 41}],
    )

    assert len(calls) == 1
    assert calls[0]["iterations"] == 3
    assert calls[0]["config"].llm.models[0].name == "gemini-3.1-flash-lite"
    assert calls[0]["config"].llm.models[0].api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert calls[0]["config"].database.random_seed == 42
    assert Path(calls[0]["initial_program"]).is_file()
    assert Path(calls[0]["evaluator"]).is_file()
    assert "OPENAI_API_KEY" not in os.environ
    assert io_to_assign
    assert io_to_cost
    assert io_to_cost != _default_cost


def test_mocked_compile_harness_runs_once_for_compile(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving()\n"
        )

    def fake_run_evolution(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ORBIT_OPENEVOLVE_POLICY_BANK", "0")
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="toy",
        openevolve_output_dir=str(tmp_path),
    )
    graph = _branch_merge_pdag(params)

    assign, timings = orbit_core(graph, LatencyEstimator(params), params)

    assert assign is not None
    assert assign.check_assign() is True
    assert len(calls) == 1
    assert calls[0]["iterations"] == 2
    assert Path(calls[0]["initial_program"]).is_file()
    assert Path(calls[0]["evaluator"]).is_file()
    assert "OpenEvolve Compile Harness Time" in timings


def test_compile_harness_fail_open_returns_initial_seed(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    def fake_run_evolution(**_kwargs):
        raise TimeoutError("llm timed out")

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ORBIT_OPENEVOLVE_POLICY_BANK", "0")
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="toy",
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["strategy"] == "level_preserving"
    assert hints["refresh_fanout_at_level_floor"] is True
    assert hints["max_scale_candidates"] == 10
    assert (tmp_path / "compile_oe_toy" / "openevolve_recovery.json").is_file()


def test_compile_harness_fail_open_recovers_checkpoint_candidate(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    code = (
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).level_preserving(level_drop_penalty=321.0)\n"
    )

    def fake_run_evolution(**kwargs):
        program_dir = Path(kwargs["output_dir"]) / "checkpoints" / "checkpoint_1" / "programs"
        program_dir.mkdir(parents=True)
        (program_dir / "candidate.json").write_text(
            json.dumps({"code": code, "metrics": {"combined_score": 2.0}}),
            encoding="utf-8",
        )
        raise TimeoutError("llm timed out")

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 321.0 else 100.0,
            "bootstrap_count": 1,
            "rescale_count": 1,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="toy",
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 321.0


def test_compile_harness_reruns_best_candidate_on_full_bundle(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving(level_drop_penalty=123.0)\n"
        )

    def fake_run_evolution(**_kwargs):
        return FakeResult()

    eval_suites = []

    def fake_evaluate_compile_hints(context, hints, *, suppress_output):
        eval_suites.append((context.get("harness", {}).get("eval_suite"), hints.get("level_drop_penalty")))
        is_evolved = hints.get("level_drop_penalty") == 123.0
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 100.0 if is_evolved else 120.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {},
            "bootstrap_locations": {},
            "diagnostics": {},
            "log_tail": "",
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        openevolve_search_mode="legacy",
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["strategy"] == "level_preserving"
    assert hints["level_drop_penalty"] == 123.0
    assert ("polybert-sampled", 20_000_000.0) in eval_suites
    assert ("polybert-full", 123.0) in eval_suites
    assert (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").is_file()


def test_full_bundle_finalist_prefers_lower_latency_over_bootstrap_regression(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        noise_estimator="off",
    )
    context = build_compile_context(_toy_pdag(params), params)
    initial_hints = {
        "strategy": "level_preserving",
        "level_drop_penalty": 111.0,
        "allow_seed_fallback": False,
    }
    candidate_code = (
        "def place(context):\n"
        "    return {'strategy': 'level_preserving', 'level_drop_penalty': 999.0, "
        "'allow_seed_fallback': False}\n"
    )

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_seed = hints.get("level_drop_penalty") == 111.0
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if not is_seed else 100.0,
            "bootstrap_count": 60 if not is_seed else 44,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "fallback_selected_groups": 0,
            "candidate_qbp_coverage": 1.0,
            "selected_output_state": {},
            "reserve_summary": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)

    selected = oe_backend._run_full_bundle_finalists(
        tmp_path,
        tmp_path / "openevolve_output",
        context,
        candidate_code,
        params,
        initial_hints,
    )

    summary = json.loads((tmp_path / "finalists" / "full_bundle_summary.json").read_text())
    assert selected["level_drop_penalty"] == 999.0
    assert summary["finalist_gate"]["seed_bootstrap_count"] == 44
    assert any(
        item.get("finalist_gate", {}).get("bootstrap_regression") is True
        for item in summary["candidates"]
    )
    assert summary["selected_index"] == 1


def test_full_bundle_finalist_fails_open_when_candidate_is_slower(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        noise_estimator="off",
    )
    context = build_compile_context(_toy_pdag(params), params)
    initial_hints = {
        "strategy": "level_preserving",
        "level_drop_penalty": 111.0,
        "allow_seed_fallback": False,
    }
    candidate_code = (
        "def place(context):\n"
        "    return {'strategy': 'level_preserving', 'level_drop_penalty': 999.0, "
        "'allow_seed_fallback': False}\n"
    )

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_seed = hints.get("level_drop_penalty") == 111.0
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 120.0 if not is_seed else 100.0,
            "objective_cost_usec": 120.0 if not is_seed else 100.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "fallback_selected_groups": 0,
            "candidate_qbp_coverage": 1.0,
            "selected_output_state": {},
            "reserve_summary": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)

    selected = oe_backend._run_full_bundle_finalists(
        tmp_path,
        tmp_path / "openevolve_output",
        context,
        candidate_code,
        params,
        initial_hints,
    )

    summary = json.loads((tmp_path / "finalists" / "full_bundle_summary.json").read_text())
    rejection = json.loads((tmp_path / "finalists" / "finalist_rejection_summary.json").read_text())
    assert selected["fail_open_reason"] == "zero_iteration_seed_portfolio"
    assert summary["candidates"][1]["finalist_gate"]["latency_improvement_reject"] is True
    assert rejection["reason"]["latency_improvement_reject"] is True


def test_full_bundle_finalist_dedupes_seed_equivalent_selected_path(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=2,
        openevolve_search_mode="bootstrap-mcts",
        noise_estimator="off",
    )
    context = build_compile_context(_toy_pdag(params), params)
    initial_hints = {
        "strategy": "level_preserving",
        "level_drop_penalty": 111.0,
        "allow_seed_fallback": False,
    }
    dup_code = (
        "def place(context):\n"
        "    return {'strategy': 'level_preserving', 'level_drop_penalty': 111.0, "
        "'allow_seed_fallback': False}\n"
    )
    distinct_code = (
        "def place(context):\n"
        "    return {'strategy': 'level_preserving', 'level_drop_penalty': 222.0, "
        "'max_scale_candidates': 33, 'allow_seed_fallback': False}\n"
    )

    monkeypatch.setattr(
        oe_backend,
        "_discover_finalist_codes",
        lambda _output_dir, _best_code, _limit: [dup_code, distinct_code],
    )

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_distinct = hints.get("level_drop_penalty") == 222.0
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 40.0 if is_distinct else 50.0,
            "objective_cost_usec": 40.0 if is_distinct else 50.0,
            "total_frontier_cost_usec": 40.0 if is_distinct else 50.0,
            "bootstrap_count": 9 if is_distinct else 10,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "fallback_selected_groups": 0,
            "candidate_qbp_coverage": 1.0,
            "selected_output_state": {"out_lvl": 1, "out_scl": 40},
            "reserve_summary": {},
            "assignment": {"v_lvl_out": {"x": 2 if is_distinct else 1}},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {
                "requested_boundary_groups": 1,
                "solved_boundary_groups": 1,
                "candidate_solved_boundary_groups": 1,
                "fallback_selected_boundary_groups": 0,
                "boundary_group_summaries": [
                    {
                        "group_key": {"in_lvl": 16, "in_scl": 40, "maino_v": "", "main_dag_size": 0},
                        "requested_output_levels": [1],
                        "reachable_budgets": 1,
                        "solved_budgets": 1,
                        "candidate_solved_budgets": 1,
                        "fallback_selected_budgets": 0,
                        "candidate_complete": True,
                        "complete": True,
                            "min_cost_usec": 40.0 if is_distinct else 50.0,
                            "min_bootstrap": 9 if is_distinct else 10,
                        "min_rescale": 2,
                        "selected_source_counts": {"candidate:unit-test": 1},
                    }
                ],
            },
            "log_tail": "",
        }

    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)

    selected = oe_backend._run_full_bundle_finalists(
        tmp_path,
        tmp_path / "openevolve_output",
        context,
        dup_code,
        params,
        initial_hints,
    )

    summary = json.loads((tmp_path / "finalists" / "full_bundle_summary.json").read_text())
    assert selected["level_drop_penalty"] == 222.0
    assert summary["selected_index"] == 2
    assert summary["effective_dedupe"]["duplicate_candidates"] >= 1
    assert summary["candidates"][1]["effective_duplicate_of_index"] == 0


def test_compile_harness_reuses_existing_output_without_llm(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "compile_oe_toy"
    best_dir = root / "openevolve_output" / "best"
    best_dir.mkdir(parents=True)
    best_dir.joinpath("best_program.py").write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).level_preserving(level_drop_penalty=456.0)\n",
        encoding="utf-8",
    )

    def fake_run_evolution(**_kwargs):
        raise AssertionError("reuse mode must not call OpenEvolve")

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 456.0 else 100.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=50,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_search_mode="legacy",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
        openevolve_reuse_output=True,
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 456.0
    assert (root / "finalists" / "full_bundle_summary.json").is_file()
    assert (root / "finalists" / "full_bundle_progress.json").is_file()


def test_noise_estimator_invalid_finalist_falls_back_to_initial_seed(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving(level_drop_penalty=123.0)\n"
        )

    def fake_run_evolution(**_kwargs):
        return FakeResult()

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 123.0 else 100.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    def fake_noise(_context, result, _params):
        return {
            "valid": False,
            "fallback": False,
            "estimated_precision_bits": 1.0,
            "output_margin_bits": -1.0,
            "unsupported_ops": [],
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    monkeypatch.setattr(oe_backend, "estimate_compile_result_noise", fake_noise)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_search_mode="legacy",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["strategy"] == "level_preserving"
    assert hints.get("level_drop_penalty") != 123.0
    summary = json.loads(
        (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").read_text()
    )
    assert summary["candidates"][0]["noise_estimator"]["valid"] is False


def test_noise_estimator_off_allows_fast_finalist(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving(level_drop_penalty=123.0)\n"
        )

    _install_fake_openevolve(monkeypatch, lambda **_kwargs: FakeResult())
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        oe_backend,
        "_evaluate_compile_hints",
        lambda _context, hints, *, suppress_output: {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 123.0 else 100.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        },
    )
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 123.0


def test_full_bundle_finalist_gate_ignores_legacy_min_bootstrap_count(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving(level_drop_penalty=123.0)\n"
        )

    def fake_run_evolution(**_kwargs):
        return FakeResult()

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_fast_low_bootstrap = hints.get("level_drop_penalty") == 123.0
        is_balanced_seed = hints.get("max_scale_candidates") == 8
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if is_fast_low_bootstrap else 90.0 if is_balanced_seed else 120.0,
            "bootstrap_count": 6 if is_fast_low_bootstrap else 10,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"final_latency_usec": 100.0, "min_bootstrap_count": 10}),
        encoding="utf-8",
    )
    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
        openevolve_reference_json=str(reference),
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 123.0
    summary = json.loads(
        (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").read_text()
    )
    assert summary["finalist_gate"]["forced_bootstrap_floor"] is None
    assert summary["selected_index"] == 1


def test_full_bundle_finalist_gate_honors_explicit_forced_bootstrap_floor(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving(level_drop_penalty=123.0)\n"
        )

    def fake_run_evolution(**_kwargs):
        return FakeResult()

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_fast_low_bootstrap = hints.get("level_drop_penalty") == 123.0
        is_balanced_seed = hints.get("max_scale_candidates") == 8
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if is_fast_low_bootstrap else 90.0 if is_balanced_seed else 120.0,
            "bootstrap_count": 6 if is_fast_low_bootstrap else 10,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"final_latency_usec": 100.0, "force_min_bootstrap_count": 10}),
        encoding="utf-8",
    )
    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        openevolve_search_mode="legacy",
        openevolve_output_dir=str(tmp_path),
        openevolve_reference_json=str(reference),
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints.get("level_drop_penalty") != 123.0
    assert hints["strategy"] == "level_preserving"
    assert hints["max_scale_candidates"] == 10
    summary = json.loads(
        (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").read_text()
    )
    assert summary["finalist_gate"]["forced_bootstrap_floor"] == 10
    assert any(
        candidate.get("finalist_gate", {}).get("bootstrap_reject")
        for candidate in summary["candidates"]
        if candidate.get("label") == "candidate_0"
    )


def test_plaintext_quality_gate_rejects_near_constant_reference():
    reject, reason = oe_backend._finalist_plaintext_quality_reject(
        {
            "harness": {
                "reference_json": {
                    "plaintext_sanity": {
                        "records": 10,
                        "near_constant_first2_logits": True,
                        "first2_margin_min": 0.0001,
                    },
                    "min_plaintext_logit_margin": 0.001,
                }
            }
        }
    )
    assert reject is True
    assert reason == "near_constant_first2_logits"


def test_profile_noise_metadata_gate_requires_estimator_profile():
    reject, reason = oe_backend._profile_noise_metadata_reject(
        {"resilience": {"enabled": True, "ckks_noise_model": None}}
    )
    assert reject is True
    assert reason == "missing_ckks_noise_model"
    reject, reason = oe_backend._profile_noise_metadata_reject(
        {
            "resilience": {
                "enabled": True,
                "ckks_noise_model": {"model": "tuneinsight-lattigo-v6", "fallback": False},
            }
        }
    )
    assert reject is False
    assert reason is None


def test_balanced_noise_margin_is_not_tied_to_reference_bootstrap_count(
    toy_cost_json: str,
    tmp_path: Path,
):
    params = _params(toy_cost_json, openevolve_reference_json=str(tmp_path / "reference.json"))
    (tmp_path / "reference.json").write_text(
        json.dumps({"min_bootstrap_count": 11, "force_min_bootstrap_count": 11}),
        encoding="utf-8",
    )
    context = build_compile_context(_toy_pdag(params), params)

    policy = PlacementBuilder(context).balanced_noise_margin()["policy"]

    assert policy["max_scale_candidates"] == 8


def test_ilp_style_reserve_metrics_are_exposed_to_openevolve(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)
    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        PlacementBuilder(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params))
        .noise_guarded_refresh()
        ["policy"],
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    reserve = oe_backend._assignment_reserve_summary(assign, graph, params)

    assert reserve["min_decryptability_reserve_bits"] is not None
    assert reserve["min_transition_reserve_bits"] is not None
    assert reserve["min_decryptability_reserve_bits"] >= 0
    assert reserve["min_transition_reserve_bits"] >= 0
    assert oe_backend._reserve_quality_score(reserve) >= 0.0


def test_estimator_backed_profile_rejects_trace_noise_warnings(toy_cost_json: str):
    params = _params(toy_cost_json)
    noise = {
        "valid": True,
        "warning_ops": ["trace_precision_below_margin"],
    }

    reject, reason = oe_backend._finalist_noise_warning_reject(
        {
            "resilience": {
                "enabled": True,
                "ckks_noise_model": {"model": "tuneinsight-lattigo-v6", "fallback": False},
            }
        },
        noise,
        params,
    )
    assert reject is True
    assert reason == "trace_precision_below_margin"

    reject, reason = oe_backend._finalist_noise_warning_reject(
        {"resilience": {"enabled": False}},
        noise,
        params,
    )
    assert reject is False
    assert reason is None


def test_zero_iteration_openevolve_path_does_not_import_gurobipy(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, openevolve_iterations=0)
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gurobipy" or name.startswith("gurobipy.") or name == "pulp":
            raise AssertionError("OpenEvolve placement imported an ILP solver")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    io_to_assign, io_to_cost = worker.get_qbp(graph, [{"in_lvl": -1, "in_scl": 40}])

    assert io_to_assign
    assert io_to_cost


def test_zero_iteration_worker_uses_safety_portfolio_in_legacy_mode(toy_cost_json: str, monkeypatch):
    params = _params(
        toy_cost_json,
        openevolve_iterations=0,
        openevolve_search_mode="legacy",
    )
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    captured = {}

    def fake_solve_budget_batch(_pdag, _budgets, _le, _params, hints, _diagnostics):
        captured.update(hints)
        return {}, {}

    monkeypatch.setattr(oe_backend, "solve_budget_batch", fake_solve_budget_batch)

    worker.get_qbp(graph, [{"in_lvl": -1, "in_scl": 40}])

    portfolio = captured.get("portfolio")
    assert isinstance(portfolio, list)
    assert {policy.get("min_internal_level") for policy in portfolio} >= {6, 8, 12}
    assert all(policy.get("allow_seed_fallback") is False for policy in portfolio)


def test_zero_iteration_worker_uses_bootstrap_mcts_seed_by_default(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=0)
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    captured = {}

    def fake_solve_budget_batch(_pdag, _budgets, _le, _params, hints, _diagnostics):
        captured.update(hints)
        return {}, {}

    monkeypatch.setattr(oe_backend, "solve_budget_batch", fake_solve_budget_batch)

    worker.get_qbp(graph, [{"in_lvl": -1, "in_scl": 40}])

    assert captured["strategy"] == "bootstrap_mcts"
    assert captured["mcts_rollout_budget"] >= 8
    assert captured["mcts_action_cap"] == 10
    assert captured["mcts_action_allowlist"] == [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "waterline_cost_beam",
        "profile_waterline_repair",
        "tuneinsight_avgcase_cost_beam",
        "tuneinsight_deferred_bootstrap_beam",
        "latency_mcts_repair",
        "component_budget_repair",
        "minimal_bootstrap_repair",
        "waterline_budget_repair",
    ]
    assert captured["boundary_state_cap"] == 6
    assert captured["selection_objective"] == "cost"


def test_qbp_manager_openevolve_backend_does_not_import_ilp_solvers(
    toy_cost_json: str,
):
    script = f"""
import sys
from scripts.latency_estimator.latency_estimator import LatencyEstimator
from scripts.optimizer.orbit.qbp_manager import QBPManager
from scripts.params.params import Params

params = Params(
    {toy_cost_json!r},
    "Orbit",
    "compile",
    Sw=40,
    CSw=40,
    threads=1,
    comp=False,
    part=False,
    placement_backend="openevolve",
    openevolve_iterations=0,
)
manager = QBPManager(params, LatencyEstimator(params))
assert manager.ilp_worker.__class__.__name__ == "OpenEvolvePlacementWorker"
assert "scripts.optimizer.orbit.ilp_worker" not in sys.modules
assert "scripts.optimizer.orbit.ilp_core" not in sys.modules
assert "pulp" not in sys.modules
assert not any(name == "gurobipy" or name.startswith("gurobipy.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
