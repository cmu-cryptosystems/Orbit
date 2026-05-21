#!/usr/bin/env python3
"""Run Tune Insight's CKKS noise estimator helper and export Orbit JSON.

The upstream repository keeps its Go module under ``estimator/`` while examples
import the module root. This wrapper clones the upstream repository outside the
Orbit tree, copies our helper into the upstream module's experiments directory,
and runs it there.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


UPSTREAM = "https://github.com/tuneinsight/ckks-noise-estimator.git"


def ensure_checkout(path: Path) -> Path:
    if (path / "estimator" / "go.mod").exists():
        return path
    if path.exists():
        shutil.rmtree(path)
    subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(path)], check=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path("/tmp/orbit_ckks_noise_estimator"))
    parser.add_argument("--helper-go", type=Path, default=Path("tools/ckks_noise_estimator/main.go"))
    parser.add_argument("--out", type=Path, default=Path("profiles/ckks_noise_estimator_64k_s40.json"))
    parser.add_argument("--logN", type=int, default=16)
    parser.add_argument("--logScale", type=int, default=40)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    checkout = ensure_checkout(args.checkout.resolve())
    module_dir = checkout / "estimator"
    target_dir = module_dir / "experiments" / "orbit_export"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_go = target_dir / "orbit_export.go"
    shutil.copyfile(args.helper_go, target_go)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "go",
            "run",
            "./experiments/orbit_export",
            "-logN",
            str(args.logN),
            "-logScale",
            str(args.logScale),
            "-iters",
            str(args.iters),
            "-out",
            str(args.out.resolve()),
        ],
        cwd=module_dir,
        check=True,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
