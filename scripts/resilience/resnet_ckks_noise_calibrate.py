#!/usr/bin/env python3
"""Calibrate ResNetSiLU64k Orbit waterlines with CKKS-style noisy plaintext runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_URL = "https://raw.githubusercontent.com/corelab-src/dacapo/dacapo/examples/data/resnet20.silu.model"
BENCHMARK = "ResNetSiLU64k"
BASELINE_NAME = "orbit_ResNetSiLU64k_Lm16_Sw40_bypass_noqbp_comp_part"
NOISE_OUTPUT_SUBDIR = "orbit_noise_calibration"
DEFAULT_BASELINE_LATENCY_SEC = 1811.614

REQUIRED_ESTIMATOR_KEYS = ("add", "rotate", "mul_plain", "mul_cipher")
DEFAULT_SEARCH_MODE = "waterline"
DEFAULT_CANDIDATE_SCALES = "40,38,36,34,32,30,28,26,24,22,20,18,16,14"
DEFAULT_CONSTANT_SCALES = "40,38,36,34,32,30,28,26,24,22,20,18,16"
DEFAULT_UPSCALE_OBJECTIVE_WEIGHTS = "0,0.025,0.05,0.1,0.2,0.35,0.5"
GROUP_REGEXES = {
    "silu_core": r"SiLU_(?:poly|mul|add)",
    "residual_add": r"add\[\]layer[123]_[0-9]+_add$",
    "downsample": r"downsamp\[\]layer[23]_ds$",
    "final_pool": r"avgpool\[\]final_pool$",
    "final_linear": r"linear\[\]final_linear$",
}
EXTRA_GROUP_ORDER = ("residual_add", "downsample", "final_pool", "final_linear")

DEFAULT_OPERATION_PRECISION_BITS = {
    "constant": 60.0,
    "input": 60.0,
    "add": 46.0,
    "mul_plain": 42.0,
    "mul_cipher": 36.0,
    "rotate": 42.0,
    "rescale": 38.0,
    "upscale": 52.0,
    "bootstrap": 24.0,
    "negate": 60.0,
    "modswitch": 42.0,
}

LATENCY_RE = re.compile(r"Final tdag latency:\s*([0-9.]+)\s*sec")
ASSIGNMENT_RE = re.compile(r"Final assignment latency:\s*([0-9.]+)\s*sec")
AGGREGATE_RE = re.compile(r"Final cost \(aggregated partition cost\):\s*([0-9.]+)\s*sec")
OP_COUNT_RE = re.compile(r"^\s+([a-z_]+):\s*([0-9.]+)\s*ops\.", re.MULTILINE)
OP_LAT_RE = re.compile(r"^\s+([a-z_]+):\s*([0-9.]+)\s*sec\.", re.MULTILINE)


@dataclass(frozen=True)
class Artifacts:
    constants: Path
    inputs_dir: Path
    plrefs_dir: Path
    true_labels: Path

    def input_file(self, sample: int) -> Path:
        return self.inputs_dir / f"input{sample}.txt"

    def plref_file(self, sample: int) -> Path:
        return self.plrefs_dir / f"plref{sample}.txt"


@dataclass(frozen=True)
class BackendRun:
    label: int
    logits: list[float]
    output_path: Path
    log_path: Path
    err_path: Path
    noise_report_path: Path | None


class NonFiniteLogitsError(RuntimeError):
    def __init__(self, path: Path, logits: list[float], bad_indices: list[int]) -> None:
        self.path = path
        self.logits = logits
        self.bad_indices = bad_indices
        super().__init__(
            f"output file {path} contains non-finite logits at indices {bad_indices}"
        )


def repo_path(repo_root: Path, path: Path | str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def run_logged(cmd: list[str], cwd: Path, log_path: Path, err_path: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if err_path is not None:
        err_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stdout:
        stderr_target = subprocess.STDOUT
        stderr_handle = None
        if err_path is not None:
            stderr_handle = err_path.open("w", encoding="utf-8")
            stderr_target = stderr_handle
        try:
            subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr_target, check=True)
        finally:
            if stderr_handle is not None:
                stderr_handle.close()


def load_operation_precision(path: Path | None) -> dict[str, float]:
    precision = dict(DEFAULT_OPERATION_PRECISION_BITS)
    if path is None or not path.exists():
        return precision
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("operation_precision_bits", data)
    for key, value in values.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            precision[str(key)] = parsed
    return precision


def ensure_estimator(repo_root: Path, estimator_json: Path) -> dict[str, Any]:
    if not estimator_json.exists():
        cmd = [
            sys.executable,
            "-m",
            "scripts.resilience.export_ckks_noise_estimator",
            "--out",
            str(estimator_json),
        ]
        run_logged(
            cmd,
            repo_root,
            estimator_json.with_suffix(".export.log"),
            estimator_json.with_suffix(".export.err"),
        )
    data = json.loads(estimator_json.read_text(encoding="utf-8"))
    precision = load_operation_precision(estimator_json)
    missing = [key for key in REQUIRED_ESTIMATOR_KEYS if precision.get(key, 0.0) <= 0.0]
    if missing:
        raise RuntimeError(f"noise estimator did not produce positive precision bits for: {', '.join(missing)}")
    return data


def ensure_checkpoint(repo_root: Path, data_root: Path) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    checkpoint = data_root / "resnet20.silu.model"
    if not checkpoint.exists():
        tmp_path = checkpoint.with_suffix(".model.tmp")
        print(f"downloading {CHECKPOINT_URL} -> {checkpoint}")
        with urllib.request.urlopen(CHECKPOINT_URL) as response:
            tmp_path.write_bytes(response.read())
        tmp_path.replace(checkpoint)

    dacapo_data = repo_root / "frontend" / "dacapo" / "examples" / "data"
    if (repo_root / "frontend" / "dacapo").exists():
        dacapo_data.mkdir(parents=True, exist_ok=True)
        target = dacapo_data / checkpoint.name
        if not target.exists() or target.stat().st_size != checkpoint.stat().st_size:
            shutil.copy2(checkpoint, target)
    return checkpoint


def artifact_paths(repo_root: Path) -> Artifacts:
    base = repo_root / "input_data" / "64k" / "resnet" / "silu"
    return Artifacts(
        constants=repo_root / "input_constants" / f"{BENCHMARK}_hecate.cst",
        inputs_dir=base / "inputs",
        plrefs_dir=base / "plrefs",
        true_labels=base / "true_labels.txt",
    )


def missing_artifacts(artifacts: Artifacts, samples: int) -> list[Path]:
    missing: list[Path] = []
    if not artifacts.constants.exists():
        missing.append(artifacts.constants)
    if not artifacts.true_labels.exists():
        missing.append(artifacts.true_labels)
    for sample in range(samples):
        for path in (artifacts.input_file(sample), artifacts.plref_file(sample)):
            if not path.exists():
                missing.append(path)
    return missing


def ensure_dacapo_tree(repo_root: Path) -> Path:
    dacapo_dir = repo_root / "frontend" / "dacapo"
    if not dacapo_dir.exists():
        raise RuntimeError(
            "missing frontend/dacapo. Clone and patch DaCapo in this isolated worktree, "
            "or provide input_constants and input_data artifacts before running calibration."
        )
    return dacapo_dir


def generate_resnet_constants(repo_root: Path) -> None:
    dacapo_dir = ensure_dacapo_tree(repo_root)
    cmd = [
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "shopt -s expand_aliases; "
            "source config.sh; "
            "mkdir -p ../log ../../mlirs_input ../../input_constants; "
            f"hc-trace {BENCHMARK} > ../log/{BENCHMARK}_trace.txt 2> ../log/{BENCHMARK}_trace.err; "
            f"python python/propagate_comments.py examples/traced/{BENCHMARK}.mlir ../../mlirs_input/{BENCHMARK}.mlir; "
            f"cp ../../mlirs_input/{BENCHMARK}.mlir examples/traced/{BENCHMARK}.mlir; "
            f"cp examples/traced/_hecate_{BENCHMARK}.cst ../../input_constants/{BENCHMARK}_hecate.cst"
        ),
    ]
    run_logged(cmd, dacapo_dir, repo_root / "frontend" / "log" / f"{BENCHMARK}_constants.log")


def generate_resnet_samples(repo_root: Path, samples: int) -> None:
    dacapo_dir = ensure_dacapo_tree(repo_root)
    patched = repo_root / "frontend" / "dacapo_patch" / "gen_input_data.py"
    target = dacapo_dir / "examples" / "tests" / "gen_input_data.py"
    if patched.exists() and target.exists():
        shutil.copy2(patched, target)
    cmd = [
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "source config.sh; "
            f"python3 examples/tests/gen_input_data.py {samples} --model resnet --activation silu --n 64k"
        ),
    ]
    run_logged(cmd, dacapo_dir, repo_root / "frontend" / "log" / f"{BENCHMARK}_samples.log")


def ensure_artifacts(repo_root: Path, data_root: Path, samples: int) -> Artifacts:
    ensure_checkpoint(repo_root, data_root)
    artifacts = artifact_paths(repo_root)
    if artifacts.constants.exists() and all(
        artifacts.input_file(sample).exists() and artifacts.plref_file(sample).exists()
        for sample in range(samples)
    ) and artifacts.true_labels.exists():
        return artifacts

    if not artifacts.constants.exists():
        generate_resnet_constants(repo_root)
    sample_missing = any(
        not artifacts.input_file(sample).exists() or not artifacts.plref_file(sample).exists()
        for sample in range(samples)
    )
    if sample_missing or not artifacts.true_labels.exists():
        generate_resnet_samples(repo_root, samples)

    missing = missing_artifacts(artifacts, samples)
    if missing:
        display = "\n".join(str(path) for path in missing[:20])
        raise RuntimeError(f"DaCapo artifact generation did not produce required files:\n{display}")
    return artifacts


def maybe_setup_lattigo_backend(repo_root: Path, backend_dir: Path, setup_backend: bool) -> None:
    if backend_dir.exists():
        return
    if not setup_backend:
        raise RuntimeError(
            f"missing backend directory {backend_dir}. Run with --setup-backend, "
            "or clone Lattigo under backend/lattigo and apply backend/patch_lattigo.sh."
        )
    backend_root = repo_root / "backend"
    clone_dir = backend_root / "lattigo"
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/tuneinsight/lattigo.git", str(clone_dir)],
        cwd=backend_root,
        check=True,
    )
    subprocess.run(["bash", "./patch_lattigo.sh"], cwd=backend_root, check=True)
    subprocess.run(["go", "mod", "tidy"], cwd=clone_dir, check=True)


def build_fhe_binary(backend_dir: Path) -> Path:
    binary = backend_dir / "fhe_binary"
    subprocess.run(["go", "build", "-o", str(binary), "./fhe"], cwd=backend_dir, check=True)
    return binary


def parse_orbit_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"log_path": str(log_path), "missing": True}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    op_counts = {name: int(float(value)) for name, value in OP_COUNT_RE.findall(text)}
    op_latency = {name: float(value) for name, value in OP_LAT_RE.findall(text)}
    stats: dict[str, Any] = {
        "log_path": str(log_path),
        "operation_counts": op_counts,
        "operation_latency_sec": op_latency,
        "bootstrap_count": op_counts.get("bootstrap_single", 0),
        "rescale_count": op_counts.get("rescale_single", 0),
        "upscale_count": op_counts.get("upscale_single", 0),
    }
    for key, pattern in (
        ("tdag_latency_sec", LATENCY_RE),
        ("assignment_latency_sec", ASSIGNMENT_RE),
        ("aggregate_partition_cost_sec", AGGREGATE_RE),
    ):
        match = pattern.search(text)
        if match:
            stats[key] = float(match.group(1))
    return stats


def run_orbit_compile(
    repo_root: Path,
    profile: Path | None,
    output_mlir: Path,
    threads: int,
    upscale_objective_weight: float | None = None,
) -> Path:
    output_mlir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_mlir.with_suffix(".txt")
    err_path = output_mlir.with_suffix(".err")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.optimizer.orbit.optimizer",
        "--inputfile",
        f"mlirs_input/{BENCHMARK}.mlir",
        "--outputfile",
        str(output_mlir),
        "--costjson",
        "cost_models/profiled_LATTIGONEW_CPU64k_3_16.json",
        "--maxlevel",
        "16",
        "--waterscale",
        "40",
        "--threads",
        str(threads),
        "--netname",
        BENCHMARK,
    ]
    if profile is not None:
        cmd.extend(["--noise-profile", str(profile)])
    if upscale_objective_weight is not None:
        cmd.extend(["--upscale-objective-weight", str(upscale_objective_weight)])
    run_logged(cmd, repo_root, log_path, err_path)
    return log_path


def orbit_compile_artifacts_exist(output_mlir: Path) -> bool:
    return output_mlir.exists() and output_mlir.with_suffix(".txt").exists()


def build_constraint(
    group: str,
    min_scale: int,
    match_regex: str,
    constant_min_scale: int | None = None,
) -> dict[str, Any]:
    constraint: dict[str, Any] = {
        "target_id": f"noise_calibrated_{group}_s{min_scale}",
        "group": group,
        "match": {"comment_regex": match_regex},
        "min_scale": min_scale,
        "fragility_class": "ckks_noise_calibrated",
        "ports": ["in", "out"],
    }
    if constant_min_scale is not None:
        constraint["constant_min_scale"] = constant_min_scale
    return constraint


def write_profile_json(
    path: Path,
    constraints: list[dict[str, Any]],
    report: dict[str, Any],
    upscale_objective_weight: float = 0.0,
) -> None:
    profile = {
        "schema_version": "orbit-resilience-constraints-v0",
        "model": {"name": BENCHMARK},
        "objective": {"upscale_weight": upscale_objective_weight},
        "description": (
            "Noise-calibrated local waterline relaxation from backend plaintext "
            "CKKS-style operation-noise simulation. Unmatched regions retain Sw=40."
        ),
        "calibration_report": report,
        "constraints": constraints,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def write_orbit_profile(
    path: Path,
    min_scale: int,
    match_regex: str,
    report: dict[str, Any],
    constant_min_scale: int | None = None,
    group: str = "silu_core",
    upscale_objective_weight: float = 0.0,
) -> None:
    write_profile_json(
        path,
        [build_constraint(group, min_scale, match_regex, constant_min_scale)],
        report,
        upscale_objective_weight,
    )


def read_vector_file(path: Path) -> list[float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty vector file: {path}")
    expected_len = int(float(lines[0]))
    values = [float(item) for item in lines[1:]]
    if len(values) < expected_len:
        raise RuntimeError(f"vector file {path} has {len(values)} values, expected {expected_len}")
    return values[:expected_len]


def top_two_labels(logits: list[float]) -> tuple[int, int, float]:
    if len(logits) < 2:
        raise RuntimeError("at least two logits are required")
    bad_indices = [index for index, value in enumerate(logits) if not math.isfinite(value)]
    if bad_indices:
        raise ValueError(f"logits contain non-finite values at indices {bad_indices}")
    ranked = sorted(range(len(logits)), key=lambda index: logits[index], reverse=True)
    top1, runner_up = ranked[0], ranked[1]
    return top1, runner_up, logits[top1] - logits[runner_up]


def predict_label(path: Path) -> int:
    values = read_vector_file(path)
    if len(values) < 10:
        raise RuntimeError(f"output file {path} has fewer than 10 logits")
    return top_two_labels(values[:10])[0]


def logit_record(sample: int, logits: list[float], true_label: int | None = None) -> dict[str, Any]:
    top1, runner_up, margin = top_two_labels(logits[:10])
    record: dict[str, Any] = {
        "sample": sample,
        "top1": top1,
        "runner_up": runner_up,
        "top1_margin": margin,
        "logits": logits[:10],
    }
    if true_label is not None:
        record["true_label"] = true_label
        record["correct"] = top1 == true_label
    return record


def margin_for_label(logits: list[float], label: int) -> float:
    if label < 0 or label >= len(logits):
        raise ValueError(f"label {label} is out of range for {len(logits)} logits")
    other_max = max(value for index, value in enumerate(logits) if index != label)
    return logits[label] - other_max


def compare_logits(
    baseline_record: dict[str, Any],
    logits: list[float],
    output_path: Path,
    trial: int | None = None,
    seed: int | None = None,
    noise_report_path: Path | None = None,
) -> dict[str, Any]:
    baseline_logits = baseline_record["logits"]
    candidate_logits = logits[:10]
    top1, runner_up, top1_margin = top_two_labels(candidate_logits)
    baseline_top1 = int(baseline_record["top1"])
    baseline_margin = max(float(baseline_record["top1_margin"]), 1e-9)
    candidate_margin = margin_for_label(candidate_logits, baseline_top1)
    deltas = [candidate - baseline for candidate, baseline in zip(candidate_logits, baseline_logits)]
    margin_retention = max(0.0, min(1.0, candidate_margin / baseline_margin))
    record: dict[str, Any] = {
        "sample": baseline_record["sample"],
        "top1": top1,
        "runner_up": runner_up,
        "top1_margin": top1_margin,
        "baseline_top1": baseline_top1,
        "top1_matches_baseline": top1 == baseline_top1,
        "candidate_margin_for_baseline_top1": candidate_margin,
        "margin_retention": margin_retention,
        "max_abs_logit_delta": max(abs(delta) for delta in deltas),
        "l2_logit_delta": math.sqrt(sum(delta * delta for delta in deltas)),
        "logits": candidate_logits,
        "output": str(output_path),
    }
    if trial is not None:
        record["trial"] = trial
    if seed is not None:
        record["seed"] = seed
    if noise_report_path is not None:
        record["noise_report"] = str(noise_report_path)
    return record


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate_drift_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "q05_margin_retention": None,
            "min_margin_retention": None,
            "mean_margin_retention": None,
            "max_abs_logit_delta": None,
            "mean_l2_logit_delta": None,
        }
    margin_retentions = [float(record["margin_retention"]) for record in records]
    l2_deltas = [float(record["l2_logit_delta"]) for record in records]
    return {
        "count": len(records),
        "q05_margin_retention": quantile(margin_retentions, 0.05),
        "min_margin_retention": min(margin_retentions),
        "mean_margin_retention": sum(margin_retentions) / len(margin_retentions),
        "max_abs_logit_delta": max(float(record["max_abs_logit_delta"]) for record in records),
        "mean_l2_logit_delta": sum(l2_deltas) / len(l2_deltas),
    }


def read_true_labels(path: Path, samples: int) -> dict[int, int]:
    labels: dict[int, int] = {}
    if not path.exists():
        return labels
    for raw in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw:
            continue
        name, label = raw.split(":", 1)
        match = re.search(r"input(\d+)\.txt", name)
        if not match:
            continue
        sample = int(match.group(1))
        if sample < samples:
            labels[sample] = int(label.strip())
    return labels


def backend_command_prefix(fhe_binary: Path | None) -> list[str]:
    if fhe_binary is not None:
        return [str(fhe_binary)]
    return ["go", "run", "./fhe"]


def backend_run_from_output(
    output_abs: Path,
    log_path: Path,
    err_path: Path,
    noise_report_path: Path | None,
) -> BackendRun:
    logits = read_vector_file(output_abs)
    if len(logits) < 10:
        raise RuntimeError(f"output file {output_abs} has fewer than 10 logits")
    candidate_logits = logits[:10]
    bad_indices = [index for index, value in enumerate(candidate_logits) if not math.isfinite(value)]
    if bad_indices:
        raise NonFiniteLogitsError(output_abs, candidate_logits, bad_indices)
    return BackendRun(
        label=top_two_labels(candidate_logits)[0],
        logits=candidate_logits,
        output_path=output_abs,
        log_path=log_path,
        err_path=err_path,
        noise_report_path=noise_report_path,
    )


def run_backend_plain(
    backend_dir: Path,
    fhe_binary: Path | None,
    mlir: Path,
    constants: Path,
    input_file: Path,
    run_dir: Path,
    tag: str,
    estimator_json: Path | None = None,
    noise_mode: str = "off",
    noise_seed: int | None = None,
    reuse_existing: bool = False,
) -> BackendRun:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_rel = f"{NOISE_OUTPUT_SUBDIR}/{tag}.out"
    output_abs = backend_dir / "outputs" / output_rel
    log_path = run_dir / f"{tag}.log"
    err_path = run_dir / f"{tag}.err"
    noise_report_path = run_dir / f"{tag}.noise.json" if noise_mode == "simulate" else None
    if reuse_existing and output_abs.exists():
        try:
            return backend_run_from_output(output_abs, log_path, err_path, noise_report_path)
        except NonFiniteLogitsError:
            raise
        except RuntimeError:
            output_abs.unlink()
    elif output_abs.exists():
        output_abs.unlink()
    cmd = backend_command_prefix(fhe_binary) + [
        "-n",
        "65536",
        "-maxLevel",
        "16",
        "-bootstrapMinLevel",
        "3",
        "-bootstrapMaxLevel",
        "16",
        "-mlir",
        str(mlir),
        "-cons",
        str(constants),
        "-input",
        str(input_file),
        "-output",
        output_rel,
        "-heMode=false",
        "-noiseMode",
        noise_mode,
    ]
    if noise_mode == "simulate":
        if estimator_json is None:
            raise ValueError("estimator_json is required when noise_mode=simulate")
        cmd.extend(["-noiseProfile", str(estimator_json)])
        if noise_seed is not None:
            cmd.extend(["-noiseSeed", str(noise_seed)])
        if noise_report_path is not None:
            cmd.extend(["-noiseReport", str(noise_report_path)])
    run_logged(cmd, backend_dir, log_path, err_path)
    if not output_abs.exists():
        raise RuntimeError(f"backend run did not produce {output_abs}; see {log_path} and {err_path}")
    return backend_run_from_output(output_abs, log_path, err_path, noise_report_path)


def collect_clean_baseline(
    backend_dir: Path,
    fhe_binary: Path | None,
    baseline_mlir: Path,
    artifacts: Artifacts,
    run_dir: Path,
    samples: int,
    true_labels: dict[int, int],
    reuse_existing: bool = False,
    parallel_runs: int = 1,
) -> tuple[list[dict[str, Any]], float]:
    records_by_sample: dict[int, dict[str, Any]] = {}

    def run_sample(sample: int) -> tuple[int, BackendRun]:
        run = run_backend_plain(
            backend_dir,
            fhe_binary,
            baseline_mlir,
            artifacts.constants,
            artifacts.input_file(sample),
            run_dir,
            f"baseline_clean_sample{sample}",
            noise_mode="off",
            reuse_existing=reuse_existing,
        )
        return sample, run

    if parallel_runs <= 1:
        for sample in range(samples):
            sample, run = run_sample(sample)
            records_by_sample[sample] = logit_record(sample, run.logits, true_labels.get(sample))
    else:
        with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
            futures = {executor.submit(run_sample, sample): sample for sample in range(samples)}
            for future in as_completed(futures):
                sample, run = future.result()
                records_by_sample[sample] = logit_record(sample, run.logits, true_labels.get(sample))

    records = [records_by_sample[sample] for sample in range(samples)]
    correct = sum(1 for record in records if record.get("correct"))
    return records, correct / samples if samples else 0.0


def evaluate_candidate(
    backend_dir: Path,
    fhe_binary: Path | None,
    candidate_mlir: Path,
    artifacts: Artifacts,
    estimator_json: Path,
    run_dir: Path,
    scale: int,
    baseline_records: list[dict[str, Any]],
    true_labels: dict[int, int],
    trial_seeds: list[int],
    baseline_accuracy: float,
    candidate_stats: dict[str, Any],
    baseline_stats: dict[str, Any],
    tag_prefix: str | None = None,
    reuse_existing: bool = False,
    max_noisy_runs: int | None = None,
    parallel_runs: int = 1,
) -> dict[str, Any]:
    samples = len(baseline_records)
    noisy_plan = [
        (baseline_record, trial, seed)
        for baseline_record in baseline_records
        for trial, seed in enumerate(trial_seeds)
    ]
    uncapped_planned_noisy_runs = len(noisy_plan)
    if max_noisy_runs is not None and max_noisy_runs > 0:
        noisy_plan = noisy_plan[:max_noisy_runs]
    planned_noisy_runs = len(noisy_plan)
    baseline_accuracy_for_noisy_plan = (
        sum(
            1
            for baseline_record, _trial, _seed in noisy_plan
            if true_labels.get(int(baseline_record["sample"])) == int(baseline_record["top1"])
        )
        / planned_noisy_runs
        if planned_noisy_runs
        else baseline_accuracy
    )
    same_predictions = 0
    noisy_correct = 0
    total_noisy_runs = 0
    mismatches: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []
    clean_runs: list[dict[str, Any]] = []
    noisy_runs: list[dict[str, Any]] = []

    baseline_latency = float(baseline_stats.get("tdag_latency_sec", DEFAULT_BASELINE_LATENCY_SEC))
    candidate_latency = candidate_stats.get("tdag_latency_sec")
    latency_improved = candidate_latency is not None and float(candidate_latency) < baseline_latency
    if not latency_improved:
        rejection_reasons.append("latency_not_improved")

    if latency_improved:
        def run_clean_candidate(baseline_record: dict[str, Any]) -> tuple[dict[str, Any], BackendRun]:
            sample = int(baseline_record["sample"])
            prefix = tag_prefix or f"s{scale}"
            tag = f"{prefix}_clean_sample{sample}"
            run = run_backend_plain(
                backend_dir,
                fhe_binary,
                candidate_mlir,
                artifacts.constants,
                artifacts.input_file(sample),
                run_dir,
                tag,
                noise_mode="off",
                reuse_existing=reuse_existing,
            )
            return baseline_record, run

        clean_results: list[tuple[dict[str, Any], BackendRun]] = []
        if parallel_runs <= 1:
            for baseline_record in baseline_records:
                try:
                    clean_results.append(run_clean_candidate(baseline_record))
                except NonFiniteLogitsError as exc:
                    if "clean_nonfinite_logits" not in rejection_reasons:
                        rejection_reasons.append("clean_nonfinite_logits")
                    if len(mismatches) < 20:
                        mismatches.append(
                            {
                                "phase": "clean",
                                "sample": int(baseline_record["sample"]),
                                "reason": "nonfinite_logits",
                                "bad_indices": exc.bad_indices,
                                "output": str(exc.path),
                            }
                        )
                    break
        else:
            with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
                futures = {
                    executor.submit(run_clean_candidate, baseline_record): baseline_record
                    for baseline_record in baseline_records
                }
                for future in as_completed(futures):
                    baseline_record = futures[future]
                    try:
                        clean_results.append(future.result())
                    except NonFiniteLogitsError as exc:
                        if "clean_nonfinite_logits" not in rejection_reasons:
                            rejection_reasons.append("clean_nonfinite_logits")
                        if len(mismatches) < 20:
                            mismatches.append(
                                {
                                    "phase": "clean",
                                    "sample": int(baseline_record["sample"]),
                                    "reason": "nonfinite_logits",
                                    "bad_indices": exc.bad_indices,
                                    "output": str(exc.path),
                                }
                            )

        clean_results.sort(key=lambda item: int(item[0]["sample"]))
        for baseline_record, run in clean_results:
            comparison = compare_logits(baseline_record, run.logits, run.output_path)
            clean_runs.append(comparison)
            if not comparison["top1_matches_baseline"] and len(mismatches) < 20:
                mismatches.append(
                    {
                        "phase": "clean",
                        "sample": int(baseline_record["sample"]),
                        "baseline_label": baseline_record["top1"],
                        "candidate_label": comparison["top1"],
                        "output": str(run.output_path),
                    }
                )

        if any(not run["top1_matches_baseline"] for run in clean_runs):
            rejection_reasons.append("clean_top1_mismatch")

    if latency_improved and not rejection_reasons:
        def run_noisy_candidate(
            baseline_record: dict[str, Any],
            trial: int,
            seed: int,
        ) -> tuple[dict[str, Any], int, int, BackendRun]:
            sample = int(baseline_record["sample"])
            prefix = tag_prefix or f"s{scale}"
            tag = f"{prefix}_sample{sample}_trial{trial}_seed{seed}"
            run = run_backend_plain(
                backend_dir,
                fhe_binary,
                candidate_mlir,
                artifacts.constants,
                artifacts.input_file(sample),
                run_dir,
                tag,
                estimator_json=estimator_json,
                noise_mode="simulate",
                noise_seed=seed,
                reuse_existing=reuse_existing,
            )
            return baseline_record, trial, seed, run

        noisy_results: list[tuple[dict[str, Any], int, int, BackendRun]] = []
        if parallel_runs <= 1:
            for baseline_record, trial, seed in noisy_plan:
                try:
                    noisy_results.append(run_noisy_candidate(baseline_record, trial, seed))
                except NonFiniteLogitsError as exc:
                    total_noisy_runs += 1
                    if "noisy_nonfinite_logits" not in rejection_reasons:
                        rejection_reasons.append("noisy_nonfinite_logits")
                    if len(mismatches) < 20:
                        sample = int(baseline_record["sample"])
                        tag = f"{tag_prefix or f's{scale}'}_sample{sample}_trial{trial}_seed{seed}"
                        mismatches.append(
                            {
                                "phase": "noisy",
                                "sample": sample,
                                "trial": trial,
                                "seed": seed,
                                "reason": "nonfinite_logits",
                                "baseline_label": int(baseline_record["top1"]),
                                "bad_indices": exc.bad_indices,
                                "output": str(exc.path),
                                "noise_report": str(run_dir / f"{tag}.noise.json"),
                            }
                        )
        else:
            with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
                futures = {
                    executor.submit(run_noisy_candidate, baseline_record, trial, seed): (
                        baseline_record,
                        trial,
                        seed,
                    )
                    for baseline_record, trial, seed in noisy_plan
                }
                for future in as_completed(futures):
                    baseline_record, trial, seed = futures[future]
                    try:
                        noisy_results.append(future.result())
                    except NonFiniteLogitsError as exc:
                        total_noisy_runs += 1
                        if "noisy_nonfinite_logits" not in rejection_reasons:
                            rejection_reasons.append("noisy_nonfinite_logits")
                        if len(mismatches) < 20:
                            sample = int(baseline_record["sample"])
                            tag = f"{tag_prefix or f's{scale}'}_sample{sample}_trial{trial}_seed{seed}"
                            mismatches.append(
                                {
                                    "phase": "noisy",
                                    "sample": sample,
                                    "trial": trial,
                                    "seed": seed,
                                    "reason": "nonfinite_logits",
                                    "baseline_label": int(baseline_record["top1"]),
                                    "bad_indices": exc.bad_indices,
                                    "output": str(exc.path),
                                    "noise_report": str(run_dir / f"{tag}.noise.json"),
                                }
                            )

        noisy_results.sort(key=lambda item: (int(item[0]["sample"]), item[1], item[2]))
        for baseline_record, trial, seed, run in noisy_results:
            sample = int(baseline_record["sample"])
            clean_label = int(baseline_record["top1"])
            total_noisy_runs += 1
            comparison = compare_logits(
                baseline_record,
                run.logits,
                run.output_path,
                trial=trial,
                seed=seed,
                noise_report_path=run.noise_report_path,
            )
            noisy_runs.append(comparison)
            same = comparison["top1_matches_baseline"]
            same_predictions += int(same)
            noisy_correct += int(true_labels.get(sample) == comparison["top1"])
            if not same and len(mismatches) < 20:
                mismatches.append(
                    {
                        "phase": "noisy",
                        "sample": sample,
                        "trial": trial,
                        "seed": seed,
                        "baseline_label": clean_label,
                        "noisy_label": comparison["top1"],
                        "output": str(run.output_path),
                        "noise_report": str(run.noise_report_path) if run.noise_report_path else None,
                    }
                )

        if same_predictions != planned_noisy_runs and "noisy_top1_mismatch" not in rejection_reasons:
            rejection_reasons.append("noisy_top1_mismatch")

    same_rate = same_predictions / planned_noisy_runs if planned_noisy_runs and total_noisy_runs else None
    noisy_accuracy = noisy_correct / total_noisy_runs if total_noisy_runs else None
    accuracy_pass = noisy_accuracy is not None and noisy_accuracy >= baseline_accuracy_for_noisy_plan
    if total_noisy_runs and not accuracy_pass:
        rejection_reasons.append("true_label_accuracy_drop")

    clean_drift_metrics = aggregate_drift_metrics(clean_runs)
    noisy_drift_metrics = aggregate_drift_metrics(noisy_runs)
    drift_metrics = noisy_drift_metrics if noisy_runs else clean_drift_metrics
    correctness_pass = not rejection_reasons

    return {
        "scale": scale,
        "samples": samples,
        "trial_seeds": trial_seeds,
        "planned_noisy_runs": planned_noisy_runs,
        "uncapped_planned_noisy_runs": uncapped_planned_noisy_runs,
        "max_noisy_runs": max_noisy_runs,
        "total_noisy_runs": total_noisy_runs,
        "same_predictions": same_predictions,
        "same_prediction_rate": same_rate,
        "baseline_accuracy": baseline_accuracy,
        "baseline_accuracy_for_noisy_plan": baseline_accuracy_for_noisy_plan,
        "noisy_accuracy": noisy_accuracy,
        "clean_candidate_runs": clean_runs,
        "noisy_candidate_runs": noisy_runs,
        "clean_drift_metrics": clean_drift_metrics,
        "noisy_drift_metrics": noisy_drift_metrics,
        "drift_metrics": drift_metrics,
        "mismatches": mismatches,
        "rejection_reasons": rejection_reasons,
        "candidate_stats": candidate_stats,
        "latency_improved": latency_improved,
        "latency_delta_sec": (
            float(candidate_latency) - baseline_latency if candidate_latency is not None else None
        ),
        "latency_gain": (
            (baseline_latency - float(candidate_latency)) / baseline_latency
            if candidate_latency is not None and baseline_latency > 0
            else None
        ),
        "correctness_pass": correctness_pass,
        "pass": correctness_pass and latency_improved,
    }


def candidate_stability(report: dict[str, Any], drift_metric: str) -> float | None:
    if drift_metric != "margin-retention":
        raise ValueError(f"unsupported drift metric: {drift_metric}")
    value = (report.get("drift_metrics") or {}).get("q05_margin_retention")
    return float(value) if value is not None else None


def candidate_latency_gain(report: dict[str, Any], baseline_latency: float) -> float | None:
    existing = report.get("latency_gain")
    if existing is not None:
        return float(existing)
    latency = (report.get("candidate_stats") or {}).get("tdag_latency_sec")
    if latency is None or baseline_latency <= 0:
        return None
    return (baseline_latency - float(latency)) / baseline_latency


def dominates(left: dict[str, Any], right: dict[str, Any], baseline_latency: float, drift_metric: str) -> bool:
    left_gain = candidate_latency_gain(left, baseline_latency)
    right_gain = candidate_latency_gain(right, baseline_latency)
    left_stability = candidate_stability(left, drift_metric)
    right_stability = candidate_stability(right, drift_metric)
    if left_gain is None or right_gain is None or left_stability is None or right_stability is None:
        return False
    return (
        left_gain >= right_gain
        and left_stability >= right_stability
        and (left_gain > right_gain or left_stability > right_stability)
    )


def assign_pareto_ranks(
    reports: list[dict[str, Any]],
    baseline_latency: float,
    drift_metric: str,
) -> None:
    for report in reports:
        report["pareto_rank"] = None
        report["selection_latency_gain"] = candidate_latency_gain(report, baseline_latency)
        report["selection_stability"] = candidate_stability(report, drift_metric) if report.get("pass") else None

    remaining = [report for report in reports if report.get("pass")]
    rank = 1
    while remaining:
        frontier = [
            report
            for report in remaining
            if not any(
                dominates(other, report, baseline_latency, drift_metric)
                for other in remaining
                if other is not report
            )
        ]
        for report in frontier:
            report["pareto_rank"] = rank
        remaining = [report for report in remaining if report not in frontier]
        rank += 1


def frontier_entry(report: dict[str, Any], baseline_latency: float, drift_metric: str) -> dict[str, Any]:
    return {
        "candidate_id": report.get("candidate_id"),
        "latency_gain": candidate_latency_gain(report, baseline_latency),
        "stability": candidate_stability(report, drift_metric),
        "pareto_rank": report.get("pareto_rank"),
        "tdag_latency_sec": (report.get("candidate_stats") or {}).get("tdag_latency_sec"),
        "group_scales": report.get("group_scales"),
        "constant_min_scale": report.get("constant_min_scale"),
        "upscale_objective_weight": report.get("upscale_objective_weight"),
    }


def perpendicular_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        return 0.0
    x0, y0 = point
    x1, y1 = start
    x2, y2 = end
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.hypot(y2 - y1, x2 - x1)


def choose_pareto_middle(
    reports: list[dict[str, Any]],
    baseline_latency: float,
    drift_metric: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    assign_pareto_ranks(reports, baseline_latency, drift_metric)
    passing = [report for report in reports if report.get("pass")]
    frontier = [
        report
        for report in passing
        if report.get("pareto_rank") == 1
        and candidate_latency_gain(report, baseline_latency) is not None
        and candidate_stability(report, drift_metric) is not None
    ]
    frontier.sort(
        key=lambda report: (
            candidate_latency_gain(report, baseline_latency) or 0.0,
            candidate_stability(report, drift_metric) or 0.0,
        )
    )

    fastest = None
    if passing:
        fastest = max(
            passing,
            key=lambda report: (
                candidate_latency_gain(report, baseline_latency) or float("-inf"),
                candidate_stability(report, drift_metric) or float("-inf"),
            ),
        )

    chosen: dict[str, Any] | None = None
    reason = "no_passing_candidates"
    if len(frontier) == 1:
        chosen = frontier[0]
        reason = "single_frontier_candidate"
    elif len(frontier) == 2:
        stable = max(
            frontier,
            key=lambda report: (
                candidate_stability(report, drift_metric) or float("-inf"),
                candidate_latency_gain(report, baseline_latency) or float("-inf"),
            ),
        )
        other = frontier[0] if frontier[1] is stable else frontier[1]
        stable_gain = candidate_latency_gain(stable, baseline_latency) or 0.0
        other_gain = candidate_latency_gain(other, baseline_latency) or 0.0
        other_stability = candidate_stability(other, drift_metric) or 0.0
        if other_gain - stable_gain >= 0.10 and other_stability >= 0.5:
            chosen = other
            reason = "two_candidate_latency_step"
        else:
            chosen = stable
            reason = "two_candidate_stability_preferred"
    elif len(frontier) >= 3:
        start = (
            candidate_latency_gain(frontier[0], baseline_latency) or 0.0,
            candidate_stability(frontier[0], drift_metric) or 0.0,
        )
        end = (
            candidate_latency_gain(frontier[-1], baseline_latency) or 0.0,
            candidate_stability(frontier[-1], drift_metric) or 0.0,
        )
        scored: list[tuple[float, float, float, dict[str, Any]]] = []
        for report in frontier:
            point = (
                candidate_latency_gain(report, baseline_latency) or 0.0,
                candidate_stability(report, drift_metric) or 0.0,
            )
            scored.append(
                (
                    perpendicular_distance(point, start, end),
                    point[1],
                    point[0],
                    report,
                )
            )
        _, _, _, chosen = max(scored, key=lambda item: item[:3])
        reason = "pareto_knee"

    selection = {
        "policy": "pareto-middle",
        "drift_metric": drift_metric,
        "reason": reason,
        "chosen_candidate_id": chosen.get("candidate_id") if chosen is not None else None,
        "fastest_passing_candidate_id": fastest.get("candidate_id") if fastest is not None else None,
        "passing_count": len(passing),
        "rejected_count": len([report for report in reports if not report.get("pass")]),
        "frontier": [frontier_entry(report, baseline_latency, drift_metric) for report in frontier],
    }
    return chosen, selection


def select_candidate(
    reports: list[dict[str, Any]],
    baseline_latency: float,
    selection_policy: str,
    drift_metric: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if selection_policy != "pareto-middle":
        raise ValueError(f"unsupported selection policy: {selection_policy}")
    return choose_pareto_middle(reports, baseline_latency, drift_metric)


def run_remote_he_smoke(
    repo_root: Path,
    remote_host: str,
    remote_key: Path,
    remote_prefix: str,
    candidate_mlir: Path,
    artifacts: Artifacts,
    clean_output: Path,
    sample: int = 0,
) -> dict[str, Any]:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    remote_dir = f"{remote_prefix}_sample{sample}_{stamp}"
    ssh_base = ["ssh", "-i", str(remote_key), remote_host]
    scp_base = ["scp", "-i", str(remote_key)]
    subprocess.run(ssh_base + [f"mkdir -p {remote_dir}/backend {remote_dir}/artifacts"], check=True)
    subprocess.run(scp_base + [str(candidate_mlir), f"{remote_host}:{remote_dir}/artifacts/candidate.mlir"], check=True)
    subprocess.run(scp_base + [str(artifacts.constants), f"{remote_host}:{remote_dir}/artifacts/{artifacts.constants.name}"], check=True)
    subprocess.run(scp_base + [str(artifacts.input_file(sample)), f"{remote_host}:{remote_dir}/artifacts/input{sample}.txt"], check=True)
    subprocess.run(scp_base + ["-r", str(repo_root / "backend" / "lattigo_patch"), f"{remote_host}:{remote_dir}/backend/"], check=True)
    subprocess.run(scp_base + [str(repo_root / "backend" / "patch_lattigo.sh"), f"{remote_host}:{remote_dir}/backend/"], check=True)

    remote_cmd = (
        f"set -euo pipefail; cd {remote_dir}/backend; "
        "git clone --depth 1 https://github.com/tuneinsight/lattigo.git; "
        "chmod +x patch_lattigo.sh; ./patch_lattigo.sh; "
        "cd lattigo; go mod tidy; cd lowering; "
        "go run ./fhe -n 65536 -maxLevel 16 -bootstrapMinLevel 3 -bootstrapMaxLevel 16 "
        f"-mlir ../../artifacts/candidate.mlir "
        f"-cons ../../artifacts/{artifacts.constants.name} "
        f"-input ../../artifacts/input{sample}.txt -output remote_he_sample{sample}.out"
    )
    subprocess.run(ssh_base + [remote_cmd], check=True)

    local_out = repo_root / "mlirs_execute" / "noise_calibration" / f"remote_he_sample{sample}_{stamp}.out"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        scp_base + [f"{remote_host}:{remote_dir}/backend/lattigo/lowering/outputs/remote_he_sample{sample}.out", str(local_out)],
        check=True,
    )
    clean = read_vector_file(clean_output)
    he = read_vector_file(local_out)
    max_abs_diff = max(abs(a - b) for a, b in zip(clean[:10], he[:10]))
    return {
        "sample": sample,
        "remote_dir": remote_dir,
        "he_output": str(local_out),
        "clean_output": str(clean_output),
        "clean_label": max(range(10), key=lambda index: clean[index]),
        "he_label": max(range(10), key=lambda index: he[index]),
        "max_abs_diff_first_10": max_abs_diff,
    }


def parse_candidate_scales(value: str) -> list[int]:
    scales = [int(part) for part in value.split(",") if part.strip()]
    if not scales:
        raise argparse.ArgumentTypeError("at least one candidate scale is required")
    return scales


def parse_float_list(value: str) -> list[float]:
    values = [float(part) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_extra_constraint(value: str) -> dict[str, Any]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--extra-constraint must be formatted as group:min_scale:comment_regex"
        )
    group, scale_text, comment_regex = parts
    if not group.strip():
        raise argparse.ArgumentTypeError("extra constraint group must be non-empty")
    try:
        min_scale = int(scale_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("extra constraint min_scale must be an integer") from exc
    if not comment_regex:
        raise argparse.ArgumentTypeError("extra constraint comment_regex must be non-empty")
    return {
        "group": group.strip(),
        "min_scale": min_scale,
        "comment_regex": comment_regex,
    }


def parse_search_mode(value: str) -> set[str]:
    allowed = {"waterline", "const", "groups", "upscale-objective"}
    modes = {part.strip() for part in value.split(",") if part.strip()}
    unknown = modes - allowed
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown search mode(s): {', '.join(sorted(unknown))}")
    return modes or {DEFAULT_SEARCH_MODE}


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("frontend/dacapo_patch/data"))
    parser.add_argument("--estimator-json", type=Path, default=Path("profiles/ckks_noise_estimator_64k_s40.json"))
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--max-noisy-runs",
        type=int,
        default=None,
        help="Cap noisy plaintext checks per candidate; default checks samples * trials.",
    )
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=1,
        help="Number of backend plaintext checks to run concurrently.",
    )
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--candidate-scales", type=parse_candidate_scales, default=parse_candidate_scales(DEFAULT_CANDIDATE_SCALES))
    parser.add_argument("--constant-scales", type=parse_candidate_scales, default=parse_candidate_scales(DEFAULT_CONSTANT_SCALES))
    parser.add_argument("--upscale-objective-weights", type=parse_float_list, default=parse_float_list(DEFAULT_UPSCALE_OBJECTIVE_WEIGHTS))
    parser.add_argument("--correctness-gate", choices=["exact-top1"], default="exact-top1")
    parser.add_argument("--selection-policy", choices=["pareto-middle"], default="pareto-middle")
    parser.add_argument("--drift-metric", choices=["margin-retention"], default="margin-retention")
    parser.add_argument("--search-mode", type=parse_search_mode, default=parse_search_mode(DEFAULT_SEARCH_MODE))
    parser.add_argument(
        "--candidate-id-prefix",
        default="",
        help="Optional prefix for generated candidate IDs to isolate separate calibration experiments.",
    )
    parser.add_argument(
        "--extra-constraint",
        action="append",
        type=parse_extra_constraint,
        default=[],
        help="Additional high-priority profile constraint formatted as group:min_scale:comment_regex.",
    )
    parser.add_argument(
        "--stop-group-on-first-fail",
        action="store_true",
        help="During extra group relaxation, stop descending a group after the first failed relaxed floor.",
    )
    parser.add_argument("--match-regex", default=r"SiLU_(?:poly|mul|add)")
    parser.add_argument("--output-profile", type=Path, default=Path("profiles/ResNetSiLU64k_noise_calibrated.json"))
    parser.add_argument("--report-json", type=Path, default=Path("profiles/ResNetSiLU64k_noise_calibrated_report.json"))
    parser.add_argument("--run-orbit", action="store_true")
    parser.add_argument("--orbit-output-dir", type=Path, default=Path("mlirs_output/orbit/ResNet/40/SiLU/64"))
    parser.add_argument("--baseline-mlir", type=Path, default=None)
    parser.add_argument("--baseline-latency-sec", type=float, default=DEFAULT_BASELINE_LATENCY_SEC)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend-dir", type=Path, default=Path("backend/lattigo/lowering"))
    parser.add_argument("--fhe-binary", type=Path, default=None)
    parser.add_argument("--setup-backend", action="store_true")
    parser.add_argument("--build-fhe-binary", action="store_true")
    parser.add_argument("--reuse-existing-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-he-smoke", action="store_true")
    parser.add_argument("--remote-host", default="jwaters@neptune2.cmcl.cs.cmu.edu")
    parser.add_argument("--remote-key", type=Path, default=Path("~/.ssh/orbit_aws"))
    parser.add_argument("--remote-prefix", default="/home/jwaters/orbit_noise_relaxed")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.max_noisy_runs is not None and args.max_noisy_runs <= 0:
        args.max_noisy_runs = None
    if args.parallel_runs < 1:
        parser.error("--parallel-runs must be positive")

    repo_root = args.repo_root.resolve()
    data_root = repo_path(repo_root, args.data_root)
    estimator_json = repo_path(repo_root, args.estimator_json)
    orbit_output_dir = repo_path(repo_root, args.orbit_output_dir)
    output_profile = repo_path(repo_root, args.output_profile)
    report_json = repo_path(repo_root, args.report_json)
    backend_dir = repo_path(repo_root, args.backend_dir)
    fhe_binary = repo_path(repo_root, args.fhe_binary)
    assert data_root and estimator_json and orbit_output_dir and output_profile and report_json and backend_dir

    estimator_metadata = ensure_estimator(repo_root, estimator_json)
    precision_bits = load_operation_precision(estimator_json)
    ensure_checkpoint(repo_root, data_root)

    baseline_mlir = repo_path(repo_root, args.baseline_mlir) if args.baseline_mlir else orbit_output_dir / f"{BASELINE_NAME}.mlir"
    if (args.run_orbit and not (args.reuse_existing_runs and orbit_compile_artifacts_exist(baseline_mlir))) or not baseline_mlir.exists():
        print(f"compiling baseline -> {baseline_mlir}")
        run_orbit_compile(repo_root, None, baseline_mlir, args.threads)
    baseline_stats = parse_orbit_log(baseline_mlir.with_suffix(".txt"))
    baseline_stats.setdefault("tdag_latency_sec", args.baseline_latency_sec)

    trial_seeds = [args.seed + trial for trial in range(args.trials)]
    run_dir = repo_root / "mlirs_execute" / "noise_calibration"
    artifacts: Artifacts | None = None
    baseline_records: list[dict[str, Any]] = []
    baseline_predictions: list[int] = []
    baseline_accuracy = 0.0

    if not args.dry_run:
        artifacts = ensure_artifacts(repo_root, data_root, args.samples)
        maybe_setup_lattigo_backend(repo_root, backend_dir, args.setup_backend)
        if args.build_fhe_binary:
            fhe_binary = build_fhe_binary(backend_dir)
        elif fhe_binary is None and (backend_dir / "fhe_binary").exists():
            fhe_binary = backend_dir / "fhe_binary"
        true_labels = read_true_labels(artifacts.true_labels, args.samples)
        print(f"collecting clean baseline predictions for {args.samples} samples")
        baseline_records, baseline_accuracy = collect_clean_baseline(
            backend_dir,
            fhe_binary,
            baseline_mlir,
            artifacts,
            run_dir,
            args.samples,
            true_labels,
            reuse_existing=args.reuse_existing_runs,
            parallel_runs=args.parallel_runs,
        )
        baseline_predictions = [int(record["top1"]) for record in baseline_records]
    else:
        true_labels = {}

    search_modes: set[str] = args.search_mode
    results: list[dict[str, Any]] = []
    candidate_profile_dir = repo_root / "profiles" / "noise_calibration_candidates"
    baseline_latency = float(baseline_stats.get("tdag_latency_sec", args.baseline_latency_sec))

    group_order = ("silu_core",) + EXTRA_GROUP_ORDER
    state: dict[str, Any] = {
        "group_scales": {},
        "constant_scale": None,
        "upscale_weight": 0.0,
        "report": None,
        "mlir": None,
        "constraints": None,
    }

    def ordered_group_scales(group_scales: dict[str, int]) -> list[tuple[str, int]]:
        return [(group, group_scales[group]) for group in group_order if group in group_scales]

    def constraints_for(group_scales: dict[str, int], constant_scale: int | None) -> list[dict[str, Any]]:
        constraints: list[dict[str, Any]] = []
        for group, scale in ordered_group_scales(group_scales):
            match_regex = args.match_regex if group == "silu_core" else GROUP_REGEXES[group]
            constraints.append(build_constraint(group, scale, match_regex, constant_scale))
        for item in args.extra_constraint:
            constraints.append(
                build_constraint(
                    item["group"],
                    int(item["min_scale"]),
                    item["comment_regex"],
                    constant_scale,
                )
            )
        return constraints

    def candidate_id(
        stage: str,
        group_scales: dict[str, int],
        constant_scale: int | None,
        upscale_weight: float,
    ) -> str:
        parts = [args.candidate_id_prefix, stage] if args.candidate_id_prefix else [stage]
        parts.extend(f"{group}s{scale}" for group, scale in ordered_group_scales(group_scales))
        if constant_scale is not None:
            parts.append(f"c{constant_scale}")
        if upscale_weight != 0.0:
            parts.append(f"upw{upscale_weight:g}".replace(".", "p"))
        return safe_id("_".join(parts))

    def update_state(
        group_scales: dict[str, int],
        constant_scale: int | None,
        upscale_weight: float,
        report: dict[str, Any],
        candidate_mlir: Path,
        constraints: list[dict[str, Any]],
    ) -> None:
        state["group_scales"] = dict(group_scales)
        state["constant_scale"] = constant_scale
        state["upscale_weight"] = upscale_weight
        state["report"] = report
        state["mlir"] = candidate_mlir
        state["constraints"] = constraints

    def run_candidate(
        stage: str,
        group_scales: dict[str, int],
        constant_scale: int | None,
        upscale_weight: float,
    ) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
        cid = candidate_id(stage, group_scales, constant_scale, upscale_weight)
        constraints = constraints_for(group_scales, constant_scale)
        candidate_profile = candidate_profile_dir / f"{cid}.json"
        profile_metadata = {
            "candidate": True,
            "candidate_id": cid,
            "stage": stage,
            "group_scales": dict(group_scales),
            "constant_min_scale": constant_scale,
            "upscale_objective_weight": upscale_weight,
            "estimator_json": str(estimator_json),
            "trial_seeds": trial_seeds,
            "max_noisy_runs": args.max_noisy_runs,
            "parallel_runs": args.parallel_runs,
        }
        write_profile_json(candidate_profile, constraints, profile_metadata, upscale_weight)

        candidate_mlir = orbit_output_dir / f"{BASELINE_NAME}_noise_{cid}.mlir"
        should_compile = args.run_orbit or not candidate_mlir.exists()
        if args.reuse_existing_runs and orbit_compile_artifacts_exist(candidate_mlir):
            should_compile = False
        compile_error: dict[str, Any] | None = None
        if should_compile:
            print(f"compiling candidate {cid} -> {candidate_mlir}")
            try:
                run_orbit_compile(repo_root, candidate_profile, candidate_mlir, args.threads, upscale_weight)
            except subprocess.CalledProcessError as exc:
                compile_error = {
                    "returncode": exc.returncode,
                    "cmd": list(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd),
                    "log_path": str(candidate_mlir.with_suffix(".txt")),
                    "err_path": str(candidate_mlir.with_suffix(".err")),
                }
        candidate_stats = (
            parse_orbit_log(candidate_mlir.with_suffix(".txt"))
            if candidate_mlir.with_suffix(".txt").exists()
            else {
                "log_path": str(candidate_mlir.with_suffix(".txt")),
                "operation_counts": {},
                "operation_latency_sec": {},
            }
        )

        if compile_error is not None:
            candidate_latency = candidate_stats.get("tdag_latency_sec")
            report = {
                "scale": group_scales.get("silu_core"),
                "samples": len(baseline_records),
                "trial_seeds": trial_seeds,
                "planned_noisy_runs": (
                    min(len(baseline_records) * len(trial_seeds), args.max_noisy_runs)
                    if args.max_noisy_runs is not None
                    else len(baseline_records) * len(trial_seeds)
                ),
                "uncapped_planned_noisy_runs": len(baseline_records) * len(trial_seeds),
                "max_noisy_runs": args.max_noisy_runs,
                "total_noisy_runs": 0,
                "same_predictions": 0,
                "same_prediction_rate": None,
                "baseline_accuracy": baseline_accuracy,
                "noisy_accuracy": None,
                "clean_candidate_runs": [],
                "noisy_candidate_runs": [],
                "clean_drift_metrics": aggregate_drift_metrics([]),
                "noisy_drift_metrics": aggregate_drift_metrics([]),
                "drift_metrics": aggregate_drift_metrics([]),
                "mismatches": [],
                "rejection_reasons": ["compile_failed"],
                "candidate_stats": candidate_stats,
                "latency_improved": False,
                "latency_delta_sec": (
                    float(candidate_latency) - baseline_latency if candidate_latency is not None else None
                ),
                "latency_gain": (
                    (baseline_latency - float(candidate_latency)) / baseline_latency
                    if candidate_latency is not None and baseline_latency > 0
                    else None
                ),
                "correctness_pass": False,
                "pass": False,
                "candidate_mlir": str(candidate_mlir),
                "compile_error": compile_error,
            }
        elif args.dry_run:
            candidate_latency = candidate_stats.get("tdag_latency_sec")
            latency_improved = candidate_latency is not None and float(candidate_latency) < baseline_latency
            report = {
                "scale": group_scales.get("silu_core"),
                "candidate_mlir": str(candidate_mlir),
                "candidate_stats": candidate_stats,
                "latency_improved": latency_improved,
                "latency_gain": (
                    (baseline_latency - float(candidate_latency)) / baseline_latency
                    if candidate_latency is not None and baseline_latency > 0
                    else None
                ),
                "drift_metrics": aggregate_drift_metrics([]),
                "rejection_reasons": ["dry_run"],
                "pass": False,
                "dry_run": True,
            }
        else:
            assert artifacts is not None
            print(f"running noisy plaintext simulation for candidate {cid}")
            report = evaluate_candidate(
                backend_dir,
                fhe_binary,
                candidate_mlir,
                artifacts,
                estimator_json,
                run_dir,
                group_scales.get("silu_core", 0),
                baseline_records,
                true_labels,
                trial_seeds,
                baseline_accuracy,
                candidate_stats,
                baseline_stats,
                tag_prefix=cid,
                reuse_existing=args.reuse_existing_runs,
                max_noisy_runs=args.max_noisy_runs,
                parallel_runs=args.parallel_runs,
            )
            report["candidate_mlir"] = str(candidate_mlir)

        report.update(
            {
                "candidate_id": cid,
                "stage": stage,
                "group_scales": dict(group_scales),
                "constant_min_scale": constant_scale,
                "upscale_objective_weight": upscale_weight,
                "profile_path": str(candidate_profile),
                "constraints": constraints,
            }
        )
        results.append(report)
        print(
            json.dumps(
                {
                    "candidate_id": cid,
                    "stage": stage,
                    "pass": report.get("pass", False),
                    "same_prediction_rate": report.get("same_prediction_rate"),
                    "total_noisy_runs": report.get("total_noisy_runs"),
                    "planned_noisy_runs": report.get("planned_noisy_runs"),
                    "noisy_accuracy": report.get("noisy_accuracy"),
                    "q05_margin_retention": (report.get("drift_metrics") or {}).get("q05_margin_retention"),
                    "latency_gain": report.get("latency_gain"),
                    "rejection_reasons": report.get("rejection_reasons"),
                    "tdag_latency_sec": report.get("candidate_stats", {}).get("tdag_latency_sec"),
                    "bootstrap_count": report.get("candidate_stats", {}).get("bootstrap_count"),
                    "rescale_count": report.get("candidate_stats", {}).get("rescale_count"),
                    "upscale_count": report.get("candidate_stats", {}).get("upscale_count"),
                },
                sort_keys=True,
            )
        )
        return report, candidate_mlir, constraints

    def sweep_waterline() -> None:
        last_pass: tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, int]] | None = None
        for scale in args.candidate_scales:
            group_scales = {"silu_core": scale}
            report, candidate_mlir, constraints = run_candidate("waterline", group_scales, None, 0.0)
            if report.get("pass", False):
                last_pass = (report, candidate_mlir, constraints, group_scales)
        if last_pass is not None:
            report, candidate_mlir, constraints, group_scales = last_pass
            update_state(group_scales, None, 0.0, report, candidate_mlir, constraints)
        elif args.dry_run:
            fallback_group_scales = {"silu_core": args.candidate_scales[0]}
            state["group_scales"] = fallback_group_scales
            state["constraints"] = constraints_for(fallback_group_scales, None)

    def sweep_constants(stage: str) -> None:
        if not state["group_scales"]:
            return
        last_pass: tuple[dict[str, Any], Path, list[dict[str, Any]], int] | None = None
        group_scales = dict(state["group_scales"])
        upscale_weight = float(state["upscale_weight"])
        for constant_scale in args.constant_scales:
            report, candidate_mlir, constraints = run_candidate(stage, group_scales, constant_scale, upscale_weight)
            if report.get("pass", False):
                last_pass = (report, candidate_mlir, constraints, constant_scale)
        if last_pass is not None:
            report, candidate_mlir, constraints, constant_scale = last_pass
            update_state(group_scales, constant_scale, upscale_weight, report, candidate_mlir, constraints)

    def sweep_upscale_objective(stage: str) -> None:
        if not state["group_scales"]:
            return
        best_pass: tuple[dict[str, Any], Path, list[dict[str, Any]], float] | None = None
        group_scales = dict(state["group_scales"])
        constant_scale = state["constant_scale"]
        for upscale_weight in args.upscale_objective_weights:
            report, candidate_mlir, constraints = run_candidate(stage, group_scales, constant_scale, upscale_weight)
            if not report.get("pass", False):
                continue
            current_latency = report.get("candidate_stats", {}).get("tdag_latency_sec")
            best_latency = None
            if best_pass is not None:
                best_latency = best_pass[0].get("candidate_stats", {}).get("tdag_latency_sec")
            if best_pass is None or (
                current_latency is not None
                and (best_latency is None or float(current_latency) < float(best_latency))
            ):
                best_pass = (report, candidate_mlir, constraints, upscale_weight)
        if best_pass is not None:
            report, candidate_mlir, constraints, upscale_weight = best_pass
            update_state(group_scales, constant_scale, upscale_weight, report, candidate_mlir, constraints)

    def sweep_extra_groups() -> None:
        if not state["group_scales"]:
            return
        for group in EXTRA_GROUP_ORDER:
            last_pass: tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, int]] | None = None
            base_group_scales = dict(state["group_scales"])
            for scale in args.candidate_scales:
                trial_group_scales = dict(base_group_scales)
                trial_group_scales[group] = scale
                report, candidate_mlir, constraints = run_candidate(
                    f"group_{group}",
                    trial_group_scales,
                    state["constant_scale"],
                    float(state["upscale_weight"]),
                )
                if report.get("pass", False):
                    last_pass = (report, candidate_mlir, constraints, trial_group_scales)
                elif args.stop_group_on_first_fail and (
                    last_pass is not None or scale == args.candidate_scales[0]
                ):
                    print(
                        json.dumps(
                            {
                                "group": group,
                                "stopped_after_failed_scale": scale,
                                "reason": "stop_group_on_first_fail",
                            },
                            sort_keys=True,
                        )
                    )
                    break
            if last_pass is None:
                continue
            report, candidate_mlir, constraints, accepted_group_scales = last_pass
            update_state(
                accepted_group_scales,
                state["constant_scale"],
                float(state["upscale_weight"]),
                report,
                candidate_mlir,
                constraints,
            )
            if accepted_group_scales.get(group) == args.candidate_scales[0]:
                print(
                    json.dumps(
                        {
                            "group": group,
                            "reason": "group_floor_remained_at_baseline",
                            "skipped_group_refinement": True,
                        },
                        sort_keys=True,
                    )
                )
                continue
            if "const" in search_modes:
                sweep_constants(f"group_{group}_const")
            if "upscale-objective" in search_modes:
                sweep_upscale_objective(f"group_{group}_upscale_objective")

    if "waterline" in search_modes:
        sweep_waterline()
    else:
        fallback_group_scales = {"silu_core": args.candidate_scales[0]}
        fallback_constraints = constraints_for(fallback_group_scales, None)
        state["group_scales"] = fallback_group_scales
        state["constraints"] = fallback_constraints

    if "const" in search_modes:
        sweep_constants("const")
    if "upscale-objective" in search_modes:
        sweep_upscale_objective("upscale_objective")
    if "groups" in search_modes:
        sweep_extra_groups()

    accepted_report, selection = select_candidate(
        results,
        baseline_latency,
        args.selection_policy,
        args.drift_metric,
    )
    accepted_scale = accepted_report.get("group_scales", {}).get("silu_core") if accepted_report is not None else None
    accepted_mlir = Path(accepted_report["candidate_mlir"]) if accepted_report is not None else None
    accepted_constraints = accepted_report.get("constraints") if accepted_report is not None else None
    accepted_upscale_weight = (
        float(accepted_report.get("upscale_objective_weight", 0.0))
        if accepted_report is not None
        else 0.0
    )

    accepted_stats = accepted_report.get("candidate_stats", {}) if isinstance(accepted_report, dict) else {}
    accepted_latency = accepted_stats.get("tdag_latency_sec")
    latency_summary = None
    if accepted_latency is not None:
        latency_summary = {
            "baseline_tdag_latency_sec": baseline_latency,
            "candidate_tdag_latency_sec": accepted_latency,
            "latency_delta_sec": float(accepted_latency) - baseline_latency,
            "bootstrap_delta": int(accepted_stats.get("bootstrap_count", 0))
            - int(baseline_stats.get("bootstrap_count", 0)),
            "rescale_delta": int(accepted_stats.get("rescale_count", 0))
            - int(baseline_stats.get("rescale_count", 0)),
            "upscale_delta": int(accepted_stats.get("upscale_count", 0))
            - int(baseline_stats.get("upscale_count", 0)),
        }

    final_report: dict[str, Any] = {
        "benchmark": BENCHMARK,
        "checkpoint_url": CHECKPOINT_URL,
        "search_mode": sorted(search_modes),
        "candidate_scales": args.candidate_scales,
        "constant_scales": args.constant_scales,
        "upscale_objective_weights": args.upscale_objective_weights,
        "correctness_gate": args.correctness_gate,
        "selection_policy": args.selection_policy,
        "drift_metric": args.drift_metric,
        "estimator_json": str(estimator_json),
        "estimator_metadata": estimator_metadata,
        "operation_precision_bits": precision_bits,
        "baseline_mlir": str(baseline_mlir),
        "baseline_stats": baseline_stats,
        "baseline_logits": baseline_records,
        "baseline_predictions": baseline_predictions,
        "baseline_accuracy": baseline_accuracy,
        "accepted_scale": accepted_scale,
        "accepted_group_scales": accepted_report.get("group_scales") if accepted_report is not None else None,
        "accepted_constant_scale": accepted_report.get("constant_min_scale") if accepted_report is not None else None,
        "accepted_upscale_objective_weight": accepted_upscale_weight if accepted_report is not None else None,
        "accepted_constraints": accepted_constraints if accepted_report is not None else None,
        "latency_summary": latency_summary,
        "selection": selection,
        "accepted_report": accepted_report,
        "candidate_results": results,
        "criteria": {
            "samples": args.samples,
            "trials": args.trials,
            "trial_seeds": trial_seeds,
            "max_noisy_runs": args.max_noisy_runs,
            "parallel_runs": args.parallel_runs,
            "correctness_gate": args.correctness_gate,
            "min_same_prediction_rate": 1.0,
            "require_clean_candidate_top1_match": True,
            "require_no_true_accuracy_drop": True,
            "require_latency_improvement": True,
            "drift_metric": args.drift_metric,
        },
    }

    if args.remote_he_smoke and accepted_mlir is not None and artifacts is not None and accepted_report is not None:
        smoke_samples = {0}
        selected_clean_runs = accepted_report.get("clean_candidate_runs") or []
        if selected_clean_runs:
            lowest_margin = min(selected_clean_runs, key=lambda record: float(record["top1_margin"]))
            smoke_samples.add(int(lowest_margin["sample"]))
        elif baseline_records:
            lowest_margin = min(baseline_records, key=lambda record: float(record["top1_margin"]))
            smoke_samples.add(int(lowest_margin["sample"]))
        final_report["remote_he_smoke"] = []
        for sample in sorted(smoke_samples):
            print(f"running remote HE smoke test for accepted candidate sample {sample}")
            clean_run = run_backend_plain(
                backend_dir,
                fhe_binary,
                accepted_mlir,
                artifacts.constants,
                artifacts.input_file(sample),
                run_dir,
                f"accepted_{accepted_report.get('candidate_id', accepted_scale)}_clean_sample{sample}",
                noise_mode="off",
                reuse_existing=args.reuse_existing_runs,
            )
            final_report["remote_he_smoke"].append(
                run_remote_he_smoke(
                    repo_root,
                    args.remote_host,
                    args.remote_key.expanduser(),
                    args.remote_prefix,
                    accepted_mlir,
                    artifacts,
                    clean_run.output_path,
                    sample=sample,
                )
            )

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
    if accepted_scale is not None and accepted_constraints is not None:
        write_profile_json(output_profile, accepted_constraints, final_report, accepted_upscale_weight)
        print(f"wrote {output_profile}")
        return 0

    print("no candidate scale satisfied correctness and performance criteria", file=sys.stderr)
    return 2 if not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
