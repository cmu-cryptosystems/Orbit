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


def _strings_from_profile_layer(layer: dict[str, Any]) -> list[str]:
    values = [
        layer.get("target_id"),
        layer.get("name"),
        layer.get("module_name"),
    ]
    values.extend(layer.get("module_names") or [])
    return [str(value) for value in values if value]


def _dedupe(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _orbit_scope_aliases(module_name: str) -> list[str]:
    aliases = [module_name]
    suffix_aliases = {
        ".attention.output.LayerNorm": ".attention.output",
        ".output.LayerNorm": ".output",
        ".intermediate.dense": ".intermediate",
        ".output.dense": ".output.dense",
    }
    for suffix, replacement in suffix_aliases.items():
        if module_name.endswith(suffix):
            aliases.append(f"{module_name[: -len(suffix)]}{replacement}")
    if module_name.startswith("bert."):
        aliases.append(module_name[len("bert.") :])
    return _dedupe(aliases)


def _scope_comment_regex(scopes: list[str]) -> str | None:
    aliases = _dedupe(
        alias
        for scope in scopes
        for alias in _orbit_scope_aliases(str(scope))
    )
    if not aliases:
        return None
    alternatives = "|".join(re.escape(alias) for alias in aliases)
    return rf"(?:^|[;\s])scope=(?:{alternatives})(?:\.|$|[;\s])"


def _resolve_json_sidecar(profile_path: Path, raw_path: str | None) -> dict[str, Any] | None:
    if not raw_path:
        return None
    sidecar_path = _resolve_sidecar_path(profile_path, raw_path)
    if not sidecar_path.is_file():
        return None
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _resolve_sidecar_path(profile_path: Path, raw_path: str) -> Path:
    sidecar_path = Path(raw_path)
    if not sidecar_path.is_absolute():
        return profile_path.parent / sidecar_path
    if sidecar_path.exists():
        return sidecar_path
    # Profiles are often generated on a training host and pulled locally with
    # the sidecars under a sibling artifacts/ directory. Keep those profiles
    # relocatable without editing the JSON.
    basename = sidecar_path.name
    for candidate in (
        profile_path.parent / "artifacts" / basename,
        profile_path.parent / basename,
    ):
        if candidate.exists():
            return candidate
    return sidecar_path


def _resolve_artifacts(profile_path: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    resolved = {}
    for key, value in artifacts.items():
        if isinstance(value, str):
            artifact_path = _resolve_sidecar_path(profile_path, value)
            resolved[key] = str(artifact_path)
        else:
            resolved[key] = value
    return resolved


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
    best_plan: dict[str, Any] | None = None
    ckks_noise_model: dict[str, Any] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

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

    @classmethod
    def load(cls, path: str | Path | None) -> "ResilienceProfile | None":
        if path is None:
            return None
        profile_path = Path(path)
        raw = profile_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        constraints = []
        explicit_constraints = []

        for item in cls._iter_orbit_constraint_items(data, profile_path):
            constraint = cls._constraint_from_orbit_item(item)
            if constraint is not None:
                explicit_constraints.append(constraint)

        if explicit_constraints:
            constraints.extend(explicit_constraints)
        else:
            for layer in data.get("layers", []):
                constraint = cls._constraint_from_profiler_layer(layer)
                if constraint is not None:
                    constraints.append(constraint)

        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        model = data.get("model") or {}
        artifacts = _resolve_artifacts(profile_path, data.get("artifacts") or {})
        return cls(
            path=profile_path,
            constraints=tuple(constraints),
            schema_version=data.get("schema_version"),
            model_name=model.get("name") if isinstance(model, dict) else None,
            fingerprint=fingerprint,
            best_plan=data.get("best_plan"),
            ckks_noise_model=data.get("ckks_noise_model"),
            artifacts=artifacts,
        )

    @staticmethod
    def _iter_orbit_constraint_items(data: dict[str, Any], profile_path: Path) -> list[dict[str, Any]]:
        payloads = []
        inline = data.get("orbit_constraints")
        if isinstance(inline, dict):
            payloads.append(inline)
        elif isinstance(inline, list):
            payloads.append({"constraints": inline})
        elif isinstance(inline, str):
            sidecar = _resolve_json_sidecar(profile_path, inline)
            if sidecar is not None:
                payloads.append(sidecar)

        artifacts = data.get("artifacts") or {}
        if isinstance(artifacts, dict):
            sidecar = _resolve_json_sidecar(profile_path, artifacts.get("orbit_constraints"))
            if sidecar is not None:
                payloads.append(sidecar)

        if "constraints" in data:
            payloads.append(data)

        items = []
        for payload in payloads:
            for item in payload.get("constraints", []):
                if isinstance(item, dict):
                    items.append(item)
        return items

    @staticmethod
    def _constraint_from_orbit_item(item: dict[str, Any]) -> ResilienceConstraint | None:
        min_scale = _as_int(
            item.get("min_scale", item.get("min_log2_scale", item.get("scale_lb")))
        )
        if min_scale is None:
            return None
        match = item.get("match") or {}
        if not isinstance(match, dict):
            match = {}
        ports_value = item.get("ports", item.get("port", ("in", "out")))
        if ports_value is None:
            ports_value = ("in", "out")
        if isinstance(ports_value, str):
            ports = (ports_value,)
        else:
            ports = tuple(ports_value)
        if isinstance(item.get("port"), str):
            ports = (item["port"],)
        ports = tuple(str(port) for port in ports)
        comment_contains = item.get("comment_contains")
        if comment_contains is None:
            comment_contains = match.get("comment_contains", ())
        if comment_contains is None:
            comment_contains = ()
        comment_literal = item.get("comment", match.get("comment"))
        if comment_literal:
            if isinstance(comment_contains, str):
                comment_contains = (comment_contains,)
            comment_contains = tuple(comment_contains) + (comment_literal,)
        if isinstance(comment_contains, str):
            comment_contains = (comment_contains,)
        scopes = item.get("scope", match.get("scope", item.get("scope_aliases", ())))
        if scopes is None:
            scopes = ()
        if isinstance(scopes, str):
            scopes = [scopes]
        comment_regex = item.get("comment_regex", match.get("comment_regex"))
        if comment_regex is None:
            comment_regex = _scope_comment_regex([str(scope) for scope in scopes])
        return ResilienceConstraint(
            min_scale=min_scale,
            target_id=item.get("target_id"),
            source="orbit_constraints",
            node=item.get("node", match.get("node")),
            node_regex=item.get("node_regex", match.get("node_regex")),
            comment_contains=tuple(str(token) for token in comment_contains),
            comment_regex=comment_regex,
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
        scope_regex = _scope_comment_regex(match_tokens)
        return ResilienceConstraint(
            min_scale=min_scale,
            target_id=layer.get("target_id"),
            source="ckks_robustness_profile",
            comment_contains=tuple(match_tokens),
            comment_regex=scope_regex,
            tau_abs=tolerance.get("tau_abs"),
            fragility_class=tolerance.get("fragility_class"),
            metadata={
                "name": layer.get("name"),
                "module_names": layer.get("module_names") or [],
                "tau_rel": tolerance.get("tau_rel"),
            },
        )

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

    def relaxed_global_scale(self, default_scale: int) -> int:
        scales = [
            int(constraint.min_scale)
            for constraint in self.constraints
            if constraint.min_scale is not None and int(constraint.min_scale) > 0
        ]
        if not scales:
            return default_scale
        return min(default_scale, min(scales))

    def error_upper_bound(
        self,
        node_label: str,
        node_attrs: dict[str, Any],
        port: str,
    ) -> float | None:
        matches = [
            float(constraint.tau_abs)
            for constraint in self.matching_constraints(node_label, node_attrs, port)
            if constraint.tau_abs is not None
        ]
        if not matches:
            return None
        return min(matches)

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
            "Resilience match report: "
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
