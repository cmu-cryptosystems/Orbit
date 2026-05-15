from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.resilience.profile import ResilienceProfile


def test_profiler_schema_080_loads_orbit_constraint_sidecar(tmp_path: Path):
    sidecar = {
        "schema_version": "orbit-resilience-constraints-v0",
        "constraints": [
            {
                "target_id": "edge:bert.encoder.layer.0:out",
                "match": {
                    "comment_regex": r"(?:^|[;\s])scope=bert\.encoder\.layer\.0(?:\.|$|[;\s])"
                },
                "min_scale": 28,
                "tau_abs": 0.001,
                "fragility_class": "sensitive",
                "ports": ["in", "out"],
            }
        ],
    }
    sidecar_path = tmp_path / "profile.orbit_constraints.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    profile_payload = {
        "schema_version": "0.8.0",
        "model": {"name": "bert-smoke"},
        "artifacts": {
            "orbit_constraints": sidecar_path.name,
            "state_traces": "profile.state_traces.jsonl",
        },
        "best_plan": {
            "action_counts": {"bootstrap": 1},
            "expected_latency": 123.0,
        },
        "layers": [
            {
                "target_id": "edge:bert.encoder.layer.0:out",
                "name": "bert.encoder.layer.0",
                "module_names": ["bert.encoder.layer.0"],
                "tolerance_profile": {
                    "tau_abs": 0.002,
                    "required_precision_bits_q99": 18,
                    "fragility_class": "fallback",
                },
                "ckks_hint": {"min_log2_scale": 18},
            }
        ],
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile_payload), encoding="utf-8")

    profile = ResilienceProfile.load(profile_path)

    assert profile is not None
    assert profile.schema_version == "0.8.0"
    assert profile.model_name == "bert-smoke"
    assert profile.best_plan == profile_payload["best_plan"]
    assert profile.artifacts["orbit_constraints"] == str(sidecar_path)
    assert len(profile.constraints) == 1
    assert profile.scale_lower_bound(
        "node0",
        {"comment": "scope=bert.encoder.layer.0.attention.self.query;op=linear"},
        20,
        "in",
    ) == 28
    assert profile.scale_lower_bound(
        "node10",
        {"comment": "scope=bert.encoder.layer.10.attention.self.query;op=linear"},
        20,
        "in",
    ) == 20


def test_orbit_native_constraints_support_node_and_scope_matching(tmp_path: Path):
    payload = {
        "schema_version": "orbit-resilience-constraints-v0",
        "constraints": [
            {"node": "5", "min_scale": 44, "port": "out"},
            {
                "scope": "bert.encoder.layer.0.attention.output.LayerNorm",
                "min_scale": 31,
                "ports": ["in"],
            },
        ],
    }
    path = tmp_path / "constraints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = ResilienceProfile.load(path)

    assert profile is not None
    assert profile.scale_lower_bound("5", {"comment": ""}, 20, "out") == 44
    assert profile.scale_lower_bound("5", {"comment": ""}, 20, "in") == 20

    layernorm_constraint = profile.constraints[1]
    assert layernorm_constraint.comment_regex is not None
    assert re.search(
        layernorm_constraint.comment_regex,
        "scope=bert.encoder.layer.0.attention.output.dense;op=linear;layer=0",
    )
