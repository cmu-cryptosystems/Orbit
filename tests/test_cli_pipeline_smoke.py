"""Fast CLI orchestration tests for Orbit pipeline scripts."""

from __future__ import annotations

import runpy
import re
import subprocess
import sys
from pathlib import Path

import pytest


def test_run_orbit_builds_expected_subprocess_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None):
            captured["cmd"] = cmd
            captured["stdout_name"] = getattr(stdout, "name", "")
            captured["stderr_name"] = getattr(stderr, "name", "")

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_orbit.py",
            "--model",
            "ResNet",
            "--act",
            "SiLU",
            "--n",
            "64",
            "--Lm",
            "16",
            "--Sw",
            "40",
            "--Csw",
            "41",
            "--nobypass",
            "--qbp",
            "--nocomp",
            "--nopart",
            "--sim-vari",
        ],
    )

    runpy.run_module("scripts.optimizer.orbit.run_orbit", run_name="__main__")

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--inputfile" in cmd
    assert "mlirs_input/ResNetSiLU64k.mlir" in cmd
    assert "--costjson" in cmd
    assert "cost_models/profiled_LATTIGONEW_CPU64k_3_16_sim_vari.json" in cmd
    assert "--constantscale" in cmd
    assert "--placement-backend" in cmd
    assert "openevolve" in cmd
    assert "--openevolve-iterations" in cmd
    assert "--openevolve-provider" in cmd
    assert "--openevolve-model" in cmd
    assert "gemini-3.1-flash-lite" in cmd
    assert "--openevolve-api-key-env" in cmd
    assert "--openevolve-llm-timeout-sec" in cmd
    assert "--openevolve-fail-open" in cmd
    assert "--noise-estimator" in cmd
    assert "--nobypass" in cmd
    assert "--enable-reqbp" in cmd
    assert "--no-compress" in cmd
    assert "--no-partition" in cmd
    assert str(captured["stdout_name"]).endswith(".txt")
    assert str(captured["stderr_name"]).endswith(".err")


def test_run_orbit_rejects_invalid_sim_vari_combo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_orbit.py",
            "--model",
            "ResNet",
            "--act",
            "SiLU",
            "--n",
            "16",
            "--Lm",
            "16",
            "--Sw",
            "40",
            "--sim-vari",
        ],
    )

    with pytest.raises(AssertionError):
        runpy.run_module("scripts.optimizer.orbit.run_orbit", run_name="__main__")


def test_optimizer_main_wires_cli_flags_to_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import scripts.optimizer.orbit.optimizer as optimizer_mod

    captured: dict[str, object] = {}

    class FakeParams:
        def __init__(self, costjson, sysname, mode, **kwargs):
            captured["init"] = {
                "costjson": costjson,
                "sysname": sysname,
                "mode": mode,
                **kwargs,
            }
            self.lvl_ub = None
            self.bts_ub = None
            self.bts_lb = None
            self.Sf = None

    def fake_run(input_file, output_file, params):
        captured["run"] = {
            "input_file": input_file,
            "output_file": output_file,
            "params": params,
        }

    def fake_makedirs(path, exist_ok=False):
        captured["makedirs"] = {"path": path, "exist_ok": exist_ok}

    monkeypatch.setattr(optimizer_mod, "Params", FakeParams)
    monkeypatch.setattr(optimizer_mod, "run", fake_run)
    monkeypatch.setattr(optimizer_mod.os, "makedirs", fake_makedirs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimizer.py",
            "--inputfile",
            "mlirs_input/motivation.mlir",
            "--outputfile",
            str(tmp_path / "out" / "final.mlir"),
            "--costjson",
            "cost_models/toy_backend.json",
            "--maxlevel",
            "15",
            "--btsupperbound",
            "13",
            "--btslevel",
            "2",
            "--waterscale",
            "40",
            "--constantscale",
            "39",
            "--rescale",
            "51",
            "--nobypass",
            "--no-compress",
            "--no-partition",
            "--enable-reqbp",
            "--threads",
            "2",
            "--ilp-solver",
            "pulp",
            "--placement-backend",
            "openevolve",
            "--openevolve-iterations",
            "0",
            "--openevolve-seed",
            "7",
            "--openevolve-provider",
            "gemini",
            "--openevolve-model",
            "gemini-3.1-pro-preview",
            "--openevolve-api-key-env",
            "GEMINI_API_KEY",
        ],
    )

    optimizer_mod.main()

    init = captured["init"]
    assert init["costjson"] == "cost_models/toy_backend.json"
    assert init["mode"] == "compile"
    assert init["Sw"] == 40
    assert init["CSw"] == 39
    assert init["bpsdepth"] is None
    assert init["threads"] == 2
    assert init["comp"] is False
    assert init["part"] is False
    assert init["reqbp"] is True
    assert init["netname"] == "motivation"
    assert init["ilp_solver"] == "pulp"
    assert init["placement_backend"] == "openevolve"
    assert init["openevolve_iterations"] == 0
    assert init["openevolve_seed"] == 7
    assert init["openevolve_provider"] == "gemini"
    assert init["openevolve_model"] == "gemini-3.1-pro-preview"
    assert init["openevolve_api_key_env"] == "GEMINI_API_KEY"
    assert init["openevolve_llm_timeout_sec"] == 180
    assert init["openevolve_llm_retries"] == 1
    assert init["openevolve_llm_retry_delay_sec"] == 2
    assert init["openevolve_evaluator_timeout_sec"] == 180
    assert init["openevolve_parallel_evaluations"] == 1
    assert init["openevolve_checkpoint_interval"] == 5
    assert init["openevolve_fail_open"] is True
    assert init["noise_estimator"] == "finalists"
    assert init["noise_estimator_timeout_sec"] == 30
    assert init["noise_estimator_min_output_margin_bits"] == 2.0

    run_call = captured["run"]
    assert run_call["input_file"] == "mlirs_input/motivation.mlir"
    assert run_call["output_file"] == str(tmp_path / "out" / "final.mlir")
    params_obj = run_call["params"]
    assert params_obj.lvl_ub == 15
    assert params_obj.bts_ub == 13
    assert params_obj.bts_lb == 2
    assert params_obj.Sf == 51

    mkdir_call = captured["makedirs"]
    assert mkdir_call["path"] == str(tmp_path / "out")
    assert mkdir_call["exist_ok"] is True


def test_optimizer_cli_openevolve_resilience_smoke(
    orbit_root: Path,
    motivation_mlir: str,
    toy_cost_json: str,
    tmp_path: Path,
):
    output_path = tmp_path / "out.mlir"
    profile_path = orbit_root / "examples" / "resilience_constraints_example.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimizer.orbit.optimizer",
            "--inputfile",
            motivation_mlir,
            "--outputfile",
            str(output_path),
            "--costjson",
            toy_cost_json,
            "--maxlevel",
            "6",
            "--waterscale",
            "40",
            "--threads",
            "2",
            "--placement-backend",
            "openevolve",
            "--openevolve-iterations",
            "0",
            "--resilience-profile",
            str(profile_path),
            "--no-compress",
            "--no-partition",
        ],
        cwd=orbit_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert output_path.is_file()
    assert "Placement backend: openevolve" in result.stdout
    assert "Resilience match report" in result.stdout

    baseline_output_path = tmp_path / "baseline.mlir"
    baseline_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimizer.orbit.optimizer",
            "--inputfile",
            motivation_mlir,
            "--outputfile",
            str(baseline_output_path),
            "--costjson",
            toy_cost_json,
            "--maxlevel",
            "6",
            "--waterscale",
            "40",
            "--threads",
            "2",
            "--placement-backend",
            "ilp",
            "--ilp-solver",
            "pulp",
            "--resilience-profile",
            str(profile_path),
            "--no-compress",
            "--no-partition",
        ],
        cwd=orbit_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert baseline_result.returncode == 0, baseline_result.stderr + baseline_result.stdout
    assert baseline_output_path.is_file()
    openevolve_bootstraps = len(re.findall(r'"earth\.bootstrap"', output_path.read_text()))
    baseline_bootstraps = len(re.findall(r'"earth\.bootstrap"', baseline_output_path.read_text()))
    assert openevolve_bootstraps <= baseline_bootstraps
