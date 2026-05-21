import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strings_from_profile_layer(layer: dict[str, Any]) -> list[str]:
    values = [
        layer.get("target_id"),
        layer.get("name"),
        layer.get("module_name"),
    ]
    values.extend(layer.get("module_names") or [])
    return [str(value) for value in values if value]


def _layer_min_scale(layer: dict[str, Any]) -> int | None:
    ckks_hint = layer.get("ckks_hint") or {}
    tolerance = layer.get("tolerance_profile") or {}
    candidates = [
        ckks_hint.get("min_log2_scale"),
        ckks_hint.get("required_precision_bits_q99"),
        tolerance.get("min_bits_abs"),
        tolerance.get("required_precision_bits_q99"),
        layer.get("min_scale"),
    ]
    for candidate in candidates:
        scale = _as_int(candidate)
        if scale is not None:
            return scale
    return None


@dataclass(frozen=True)
class ResilienceConstraint:
    min_scale: int
    constant_min_scale: int | None = None
    group: str | None = None
    target_id: str | None = None
    source: str = "unknown"
    node: str | None = None
    node_regex: str | None = None
    comment_contains: tuple[str, ...] = ()
    comment_regex: str | None = None
    ports: tuple[str, ...] = ("in", "out")
    tau_abs: float | None = None
    fragility_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, node_label: str, node_attrs: dict[str, Any], port: str) -> bool:
        if self.ports and port not in self.ports:
            return False
        comment = str(node_attrs.get("comment", "") or "")
        if self.node is not None and (
            node_label == self.node or node_label.startswith(f"{self.node}_")
        ):
            return True
        if self.node_regex is not None and re.search(self.node_regex, node_label):
            return True
        if self.comment_regex is not None and re.search(self.comment_regex, comment):
            return True
        if self.comment_contains:
            return any(token and token in comment for token in self.comment_contains)
        return False


@dataclass(frozen=True)
class ResilienceProfile:
    path: Path
    constraints: tuple[ResilienceConstraint, ...]
    schema_version: str | None = None
    model_name: str | None = None
    fingerprint: str = ""
    objective_upscale_weight: float = 0.0

    @classmethod
    def load(cls, path: str | Path | None) -> "ResilienceProfile | None":
        if path is None:
            return None
        profile_path = Path(path)
        raw = profile_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        constraints = []

        for item in data.get("orbit_constraints", data.get("constraints", [])):
            constraint = cls._constraint_from_orbit_item(item)
            if constraint is not None:
                constraints.append(constraint)

        for layer in data.get("layers", []):
            constraint = cls._constraint_from_profiler_layer(layer)
            if constraint is not None:
                constraints.append(constraint)

        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        model = data.get("model") or {}
        objective = data.get("objective") or {}
        objective_upscale_weight = _as_float(objective.get("upscale_weight")) or 0.0
        return cls(
            path=profile_path,
            constraints=tuple(constraints),
            schema_version=data.get("schema_version"),
            model_name=model.get("name") if isinstance(model, dict) else None,
            fingerprint=fingerprint,
            objective_upscale_weight=objective_upscale_weight,
        )

    @staticmethod
    def _constraint_from_orbit_item(item: dict[str, Any]) -> ResilienceConstraint | None:
        min_scale = _as_int(
            item.get("min_scale", item.get("min_log2_scale", item.get("scale_lb")))
        )
        if min_scale is None:
            return None
        match = item.get("match") or {}
        ports_value = item.get("ports", item.get("port", ("in", "out")))
        if isinstance(ports_value, str):
            ports = (ports_value,)
        else:
            ports = tuple(ports_value)
        if isinstance(item.get("port"), str):
            ports = (item["port"],)
        comment_contains = item.get("comment_contains")
        if comment_contains is None:
            comment_contains = match.get("comment_contains", ())
        if isinstance(comment_contains, str):
            comment_contains = (comment_contains,)
        return ResilienceConstraint(
            min_scale=min_scale,
            constant_min_scale=_as_int(item.get("constant_min_scale")),
            group=item.get("group"),
            target_id=item.get("target_id"),
            source="orbit_constraints",
            node=item.get("node", match.get("node")),
            node_regex=item.get("node_regex", match.get("node_regex")),
            comment_contains=tuple(str(token) for token in comment_contains),
            comment_regex=item.get("comment_regex", match.get("comment_regex")),
            ports=ports,
            tau_abs=item.get("tau_abs"),
            fragility_class=item.get("fragility_class"),
            metadata={k: v for k, v in item.items() if k not in {"match"}},
        )

    @staticmethod
    def _constraint_from_profiler_layer(layer: dict[str, Any]) -> ResilienceConstraint | None:
        min_scale = _layer_min_scale(layer)
        match_tokens = _strings_from_profile_layer(layer)
        if min_scale is None or not match_tokens:
            return None
        tolerance = layer.get("tolerance_profile") or {}
        return ResilienceConstraint(
            min_scale=min_scale,
            constant_min_scale=_as_int(layer.get("constant_min_scale")),
            group=layer.get("group"),
            target_id=layer.get("target_id"),
            source="ckks_robustness_profile",
            comment_contains=tuple(match_tokens),
            tau_abs=tolerance.get("tau_abs"),
            fragility_class=tolerance.get("fragility_class"),
            metadata={
                "name": layer.get("name"),
                "module_names": layer.get("module_names") or [],
                "tau_rel": tolerance.get("tau_rel"),
            },
        )

    def matching_constraints(
        self,
        node_label: str,
        node_attrs: dict[str, Any],
        port: str,
    ) -> list[ResilienceConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if constraint.matches(node_label, node_attrs, port)
        ]

    def scale_lower_bound(
        self,
        node_label: str,
        node_attrs: dict[str, Any],
        default_scale: int,
        port: str,
    ) -> int:
        matches = [
            constraint.min_scale
            for constraint in self.matching_constraints(node_label, node_attrs, port)
        ]
        if not matches:
            return default_scale
        return max(matches)

    def constant_scale_lower_bound(
        self,
        node_label: str,
        node_attrs: dict[str, Any],
        default_constant_scale: int,
        default_cipher_scale: int,
    ) -> int:
        matches = [
            constraint.constant_min_scale
            for constraint in self.matching_constraints(node_label, node_attrs, "in")
            if constraint.constant_min_scale is not None
        ]
        if matches:
            return max(matches)
        return min(
            default_constant_scale,
            self.scale_lower_bound(node_label, node_attrs, default_cipher_scale, "in"),
        )

    def relaxed_global_scale(self, default_scale: int) -> int:
        scales = [
            int(constraint.min_scale)
            for constraint in self.constraints
            if constraint.min_scale is not None and int(constraint.min_scale) > 0
        ]
        if not scales:
            return default_scale
        return min(default_scale, min(scales))

    def match_report(self, tdag) -> dict[str, Any]:
        constraint_to_nodes: dict[int, set[str]] = {
            index: set() for index in range(len(self.constraints))
        }
        matched_nodes: set[str] = set()
        node_port_matches = 0

        for node_label in tdag.nodes:
            node_attrs = tdag.nodes[node_label]
            for index, constraint in enumerate(self.constraints):
                for port in constraint.ports or ("in", "out"):
                    if constraint.matches(node_label, node_attrs, port):
                        constraint_to_nodes[index].add(str(node_label))
                        matched_nodes.add(str(node_label))
                        node_port_matches += 1
                        break

        unmatched = [
            self.constraints[index].target_id
            or self.constraints[index].node
            or self.constraints[index].comment_regex
            or f"constraint:{index}"
            for index, nodes in constraint_to_nodes.items()
            if not nodes
        ]
        matched_constraints = sum(1 for nodes in constraint_to_nodes.values() if nodes)
        return {
            "total_constraints": len(self.constraints),
            "matched_constraints": matched_constraints,
            "matched_nodes": len(matched_nodes),
            "node_port_matches": node_port_matches,
            "unmatched_targets": unmatched,
        }

    @staticmethod
    def format_match_report(report: dict[str, Any]) -> str:
        return (
            "Noise profile match report: "
            f"constraints={report['total_constraints']}, "
            f"matched_constraints={report['matched_constraints']}, "
            f"matched_nodes={report['matched_nodes']}, "
            f"node_port_matches={report['node_port_matches']}, "
            f"unmatched_constraints={len(report['unmatched_targets'])}"
        )

    def describe(self) -> str:
        return (
            f"{self.path} ({len(self.constraints)} constraints, "
            f"fingerprint={self.fingerprint})"
        )
