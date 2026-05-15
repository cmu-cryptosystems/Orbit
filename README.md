# Orbit: Optimizing Rescale and Bootstrap Placement with OpenEvolve or ILP

## Installation

### Requirements

The following python modules are needed to run Orbit:
```
joblib
matplotlib
more-itertools
networkx
numpy
openevolve
pulp
```

Run the following commands to install them:
```bash
pip install -r requirements.txt
```

OpenEvolve is the default placement backend. `--openevolve-iterations 0` uses Orbit's deterministic conservative policy without LLM calls, which is useful for smoke tests and no-Gurobi compilation.

For positive OpenEvolve iterations, Orbit defaults to Gemini through OpenEvolve's OpenAI-compatible API support. Provide the key as an environment variable, not in tracked files:

```bash
export OPENAI_API_KEY="your-gemini-api-key"
# or, Orbit will bridge this to OPENAI_API_KEY during OpenEvolve calls:
export GEMINI_API_KEY="your-gemini-api-key"
```

Do not commit API keys. Rotate any key that has been pasted into chat, logs, shell history, or tracked files.

Gurobi is optional and only used for comparison runs with `--placement-backend ilp --ilp-solver gurobi`. Install `gurobipy` separately and provide a Gurobi license if you need that backend. The ILP comparison path also supports PuLP/CBC with `--ilp-solver pulp`.

### Frontend and Backend

Orbit uses Dacapo's frontend interface and Lattigo as backend. To evaluate the compiled MLIR files, it is necessary to install the *backend*. As the plaintext files for evaluation are too large (in GBs), it is necessary to install the *frontend* to generate such files. **If you are satisfied with estimated evaluation results, it is NOT necessary to install the frontend and backend.**

Check `frontend/README.md` and `backend/README.md` for installation instructions.

## Usage

### Compilation on one benchmark

**Requirements:**
- Input MLIR file: `mlirs_input/<model-name><activation><16|64>k.mlir`.
- Cost model: `cost_models/profiled_LATTIGONEW_CPU<16|64>k_3_<maxlevel>.json`.

**Script usage:**
```bash
# The current directory should be <orbit-repository>/
python3 scripts/optimizer/orbit/run_orbit.py [options]
```

**Options:**
- `--model <model-name>`: (Required) The model name.
- `--act <activation>`: (Required) The activation function name.
- `--n <16|64>`: (Required) The number of slots (in 1024s). 
- `--Lm <maxlevle>`: (Required) The number of effective levels for computation (i.e. $L_{bts}$ in the paper).
- `--Sw <waterline>`: (Required) The minimum scale attribute (i.e. $\log S_{min}$ in the paper).
- `--nobypass`: If enabled, the bypass detection and handling is disabled.
- `--qbp`: If enabled, Orbit will reuse qbps across different benchmarks.
- `--nocomp`: If enabled, the compression technique is disabled.
- `--nopart`: If enabled, the partitioning technique is disabled.
- `--sim-vari`: If enabled, use the cost model with simulated variadic bootstrapping cost.
- `--Csw`: Specify the minimum scale attribute for plaintexts. This option is experimental and is not used in Orbit's evaluation.
- `--placement-backend <openevolve|ilp>`: Placement backend. Default is `openevolve`.
- `--ilp-solver <pulp|gurobi>`: ILP solver, used only with `--placement-backend ilp`.
- `--openevolve-config <path>`: Optional OpenEvolve YAML config for runtime evolution.
- `--openevolve-output-dir <path>`: Directory for generated OpenEvolve workspaces/checkpoints.
- `--openevolve-iterations <n>`: OpenEvolve iterations. Use `0` for deterministic placement without LLM calls.
- `--openevolve-harness <compile|partition>`: Harness scope for positive iterations. Default `compile` runs one OpenEvolve search for the compile and replays final placement scoring; `partition` keeps the legacy per-batch behavior.
- `--openevolve-eval-suite <toy|polybert-sampled|polybert-full>`: Evaluator bundle. Default `polybert-sampled` keeps compile-level scoring fast while preserving all final validation outside the harness.
- `--openevolve-reference-json <path>`: Optional precomputed reference metrics for reporting only. This never runs PuLP/Gurobi inside OpenEvolve.
- `--openevolve-finalists <n>`: Number of candidates reserved for full-bundle finalist scoring metadata.
- `--openevolve-seed <n>`: Seed passed into the generated OpenEvolve context and config override.
- `--openevolve-provider <gemini|openai|custom>`: Provider for generated OpenEvolve config. Default is `gemini`.
- `--openevolve-model <name>`: Model for generated OpenEvolve config. Default is `gemini-3.1-flash-lite`.
- `--openevolve-api-base <url>`: OpenAI-compatible API base. Required for `--openevolve-provider custom`.
- `--openevolve-api-key-env <name>`: API key environment variable. Default is `OPENAI_API_KEY`; Gemini also falls back to `GEMINI_API_KEY`.
- `--openevolve-llm-timeout-sec <n>`, `--openevolve-llm-retries <n>`, `--openevolve-llm-retry-delay-sec <n>`: Runtime retry controls for generated OpenEvolve configs.
- `--openevolve-evaluator-timeout-sec <n>`, `--openevolve-parallel-evaluations <n>`, `--openevolve-checkpoint-interval <n>`: Evaluator and checkpoint controls for positive-iteration searches.
- `--openevolve-fail-open` / `--no-openevolve-fail-open`: Return the best recovered or initial validated candidate if OpenEvolve runtime fails. Enabled by default.
- `--openevolve-keep-workdir`: Keep temporary OpenEvolve workspaces.
- `--noise-estimator <off|finalists>`: Estimator-backed full-bundle finalist gate. Default `finalists` uses a Tune Insight/Lattigo-style CKKS precision estimate before selecting positive-iteration OpenEvolve candidates.
- `--noise-estimator-binary <path>`: Optional JSON sidecar binary for CKKS noise estimation. If omitted, Orbit uses the built-in deterministic estimator contract.
- `--noise-estimator-min-output-margin-bits <n>`: Minimum estimated precision margin for a finalist to beat a fully valid candidate. Default `2`.
- `--resilience-profile <path>`: ckks-robustness-profiler 0.8.0 profile, generated Orbit constraint sidecar, or Orbit-native constraints JSON.
- `--resilience-constraint-policy <relax-only|hard-tau>`: Use profile constraints as relaxed waterline guidance or hard local scale lower bounds.

**Output files:**

You can find the output files in `<orbit-repository>/mlirs_output/orbit/<model-name>/<waterline>/<activation>/<16|64>/`. The filename should be `orbit_<model-name><activation><16|64>k_Lm<maxlevel>_Sw<waterline>_<cmt>.<txt|err|mlir>`.
- `<cmt>` depends on the enabled/disabled techniques. For the base configuration, `<cmt>` is `bypass_noqbp_comp_part`.
- The `.txt` file is Orbit's log file. You can see the estimated execution time(`Final tdag latency`) and its breakdown, as well as Orbit's compilation time(`Orbit Compilation time`) and its breakdown at the end of the file.
- The `.err` file is Orbit's standard error output file. It should be empty.
- The `.mlir` file is Orbit's compiled MLIR file. This file is necessary for execution.

**Example:**

```bash
python3 scripts/optimizer/orbit/run_orbit.py --model ResNet --act SiLU --n 64 --Lm 16 --Sw 40
# This command is for compiling MLIR for ResNet-SiLU under base configuration (n=64k,Lm=16,Sw=40).
```

The output files can be found in `<orbit-repository>/mlirs_output/orbit/ResNet/40/SiLU/64/`. The filename is `orbit_ResNetSiLU64k_Lm16_Sw40_bypass_noqbp_comp_part`. The log file is `<filename>.txt`, the error output file is `<filename>.err`, and the compiled MLIR file is `<filename>.mlir`.

### Scripts for reproducing compilation results

You can find the scripts for reproducing the compilation results in the paper in `auto_scripts/`. Make sure the input MLIR files exist in `mlirs_input/`.
- `compile_orbit_eval_base.sh`: This script is for compiling all benchmarks on the base configuration(n=64k,Lm=16,Sw=40).
- `compile_orbit_eval_16k.sh`: This script is for compiling all benchmarks on the n=16k configuration(n=16k,Lm=16,Sw=40).
- `compile_orbit_eval_Lm12.sh`: This script is for compiling all benchmarks on the Lm=12 configuration(n=64k,Lm=12,Sw=40).
- `compile_orbit_eval_Sw51.sh`: This script is for compiling all benchmarks on the Sw=51 configuration(n=64k,Lm=16,Sw=51).
- `compile_orbit_eval_sim_vari.sh`: This script is for compiling all benchmarks on the base configuration(n=64k,Lm=16,Sw=40) using cost model with simulated variadic bootstrapping cost.
- `compile_orbit_micro_comppart.sh`: This script is for compiling the specific benchmark on the base configuration(n=64k,Lm=16,Sw=40), enabling/disabling the compression/partitioning techniques.
- `compile_orbit_micro_bypass.sh`: This script is for compiling ResNet benchmark on both base and n=16k configuration(Lm=16,Sw=40), disabling bypass handling.
- `compile_orbit_micro_reqbp.sh`: This script is for compiling all benchmarks on the n=16k configuration(n=16k,Lm=16,Sw=40), where Orbit can reuse qbp across different benchmarks.

### (Optional) Compilation on arbitrary MLIR input

If you want to compile arbitrary MLIR input file with Orbit, you need to use the frontend to generate valid MLIR input file. To compile it, use the following script: 

```bash
# The current directory should be <orbit-repository>/
python3 -m scripts.optimizer.orbit.optimizer [options]
```

Check `scripts/optimizer/orbit/optimizer.py` for options and usage.

Minimal no-Gurobi smoke compile:

```bash
python3 -m scripts.optimizer.orbit.optimizer \
  --inputfile mlirs_input/motivation.mlir \
  --outputfile /tmp/orbit_oe_smoke.mlir \
  --costjson cost_models/toy_backend.json \
  --maxlevel 6 \
  --waterscale 40 \
  --placement-backend openevolve \
  --openevolve-iterations 0 \
  --no-compress \
  --no-partition
```

Profiler-aware smoke compile:

```bash
python3 -m scripts.optimizer.orbit.optimizer \
  --inputfile mlirs_input/motivation.mlir \
  --outputfile /tmp/orbit_oe_resilience_smoke.mlir \
  --costjson cost_models/toy_backend.json \
  --maxlevel 6 \
  --waterscale 40 \
  --placement-backend openevolve \
  --openevolve-iterations 0 \
  --resilience-profile examples/resilience_constraints_example.json \
  --no-compress \
  --no-partition
```

Gemini-backed OpenEvolve search:

```bash
export OPENAI_API_KEY="your-gemini-api-key"
python3 -m scripts.optimizer.orbit.optimizer \
  --inputfile mlirs_input/motivation.mlir \
  --outputfile /tmp/orbit_oe_gemini.mlir \
  --costjson cost_models/toy_backend.json \
  --maxlevel 6 \
  --waterscale 40 \
  --placement-backend openevolve \
  --openevolve-iterations 10 \
  --openevolve-model gemini-3.1-flash-lite \
  --no-compress \
  --no-partition
```

## Overview

The structure of Orbit repository is as follows:
- `frontend/`: The frontend. It contains Orbit's patch on Dacapo.
- `backend/`: The backend. It contains Orbit's patch on Lattigo.
- `scripts/`: The core scripts of Orbit.
- `auto_scripts/`: The scripts for reproducing compilation results.
- `cost_models/`: The cost models.
- `mlirs_input/`: The MLIR input files, which were generated by Dacapo's frontend interface.
- `mlirs_output/`: (Created automatically) The compiled MLIR files and Orbit's log files.
- `mlirs_execute/`: (Created automatically) The evaluation results of MLIR files compiled by Orbit.
- `input_data/`: (Created automatically) The CIFAR-10 samples, including the input files, correct labels, and the output files generated by PyTorch models.
- `input_constants/`: (Created automatically) The plaintext files for execution.
- `qbps_cache_*/`: (Created automatically) The cached qbps for cross-benchmark qbp reusing.

### Orbit scripts

The scripts of Orbit are stored in `scripts/`, which contains the following components:
- `tdag/`: The dag structure in Orbit.
  - The automatic compression technique is implemented in `auto_compression.py`.
- `assignment/`: The assignment structure in Orbit. i.e. How Orbit stores the rescale and bootstrap placement.
- `latency_estimator/`: The latency estimator in Orbit. It can estimate the latency of a given dag or assignment, based on the parsed cost model.
- `params/`: The parameters in Orbit.
- `visualize/`: The visualization scripts for dag. Mainly for debugging.
- `utils/`: Some auxiliary scripts.
- `optimizer/orbit/`: The core scripts of Orbit:
  - The partitioning technique and bypass handling are implemented in `siso_partition.py`.
  - The iterative partition solving and merging: in `iterative_partition.py`.
  - The OpenEvolve placement backend: in `openevolve_backend.py`.
  - The optional ILP formulation: in `ilp_core.py` and `ilp_gurobi.py`.
  - The QBP management: in `qbp_manager.py`.
- `resilience/`: Loading profiler 0.8.0 and Orbit-native resilience constraints.
