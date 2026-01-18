## Backend Instructions

Orbit uses Lattigo as backend to execute FHE programs. 

### Install Lattigo

This is the link of [Lattigo](https://github.com/tuneinsight/lattigo). Please first clone the repository, and then install Orbit's patch on Lattigo:

```bash
# The current directory should be <orbit-repository>/backend/
git clone https://github.com/tuneinsight/lattigo.git
./patch_lattigo.sh
```

#### Initial Setup

1. Navigate to the lattigo root directory and ensure dependencies are installed:
```bash
# The current directory should be <orbit-repository>/backend/
cd lattigo
go mod tidy
```

2. Build the FHE binary:
```bash
# The current directory should be <orbit-repository>/backend/lattigo/
cd lowering
go build -o fhe_binary ./fhe
```

### Usage

We provide several scripts for reproducing Orbit's evaluation results. For detailed usage, please check `lattigo_patch/lowering/README.md`. 

#### Evaluating Orbit on One Benchmark, One sample

Please make sure that the corresponding sample input data and plaintext files are generated. If not, please use scripts in frontend to generate.

**Requirements:**
- Sample input file: `input_data/<16k|64k>/<model-name>/<activation>/inputs/input<id>.txt`
- MLIR plaintext file: `input_constants/<model-name><activation><16k|64k>_hecate.cst`
- Orbit's compiled MLIR file: `mlirs_output/orbit/<model-name>/<waterline-scale>/<activation>/<16|64>/orbit_*.mlir`

**Script usage:**
```bash
# The current directory should be <orbit-repository>/backend/lattigo/lowering/
python3 run_orbit_one_eval.py [options]
```

**Options:**
- `--model <model-name>`: (Required) The model name.
- `--act <activation>`: (Required) The activation function name.
- `--n <16|64>`: (Required) The number of slots (in 1024s). 
- `--Lm <maxlevle>`: (Required) The number of effective levels for computation (i.e. $L_{bts}$ in the paper).
- `--Sw <waterline>`: (Required) The minimum scale attribute (i.e. $\log S_{min}$ in the paper).
- `--run <id>`: (Required) The sample index.
- `--plain`: If enabled, execute the MLIR file in plaintext mode.
- `--cmt <orbit-comment>`: Specify the comment in the name of Orbit's compiled MLIR file. The default comment is for macro-benchmarks evaluation. Check `run_orbit_one_eval.py` for comments in micro-benchmarks evaluation.
- `--lcmt <log-comment>`: Additional comments appended to the log file.
- `--Csw`: Specify the minimum scale attribute for plaintexts. This option is experimental and is not used in Orbit's evaluation.

**Output files:**

You can find the output files in `<orbit-repository>/mlirs_execute/orbit/<16|64>/<model-name>/<activation>/`:
- The `<mlir-name>_run<id>.log` file is the execution logs for HE execution on `mlir-name` MLIR file and on the `id`-th sample.
- The `<mlir-name>_run<id>.out` file is the output vector for HE execution on `mlir-name` MLIR file and on the `id`-th sample.
- The `<mlir-name>_run<id>_pl.log` file is the execution logs for plaintext execution on `mlir-name` MLIR file and on the `id`-th sample.
- The `<mlir-name>_run<id>_pl.out` file is the output vector for plaintext execution on `mlir-name` MLIR file and on the `id`-th sample.

**Example:**
```bash
python3 run_orbit_one_eval.py --model ResNet --act SiLU --n 64 --Lm 16 --Sw 40 --run 0
# This command is for evaluating Orbit's compiled MLIR for ResNet-SiLU under base configuration (n=64k,Lm=16,Sw=40). The sample id is 0. 
```

The output files can be found in `<orbit-repository>/mlirs_execute/orbit/64/ResNet/SiLU/`, the execution log is `<mlir-name>_run0.log` and the output vector is `<mlir-name>_run0.out`.

#### Evaluating Orbit on One Benchmark, Many samples

The following script is for evaluating Orbit on the same benchmark and same configuration, but for multiple samples. It enables multi-threading to accelerate the evaluation.

**Requirements:**
- The required files for the single-sample script.
- Sample input files: `input_data/<16k|64k>/<model-name>/<activation>/inputs/input<id>.txt` for `id=0,...,number-1`.

**Script usage:**
```bash
# The current directory should be <orbit-repository>/backend/lattigo/lowering/
python3 run_orbit_one_bench_many_runs.py [options]
```

**Options:**
- `--model`, `--act`, `--n`, `--Lm`, `--Sw`, `--plain`, `--Csw`: These options are the same as those in the single-sample scripts.
- `--maxthread <num-thread>`: Set the maximum number of threads. Check `run_orbit_one_bench_many_runs.py` for recommended number of threads under different configurations.
- `--runs <number>`: Evaluate the MLIR file from sample `0` to sample `number-1`.

**Example:**
```bash
python3 run_orbit_one_bench_many_runs.py --model ResNet --act SiLU --n 64 --Lm 16 --Sw 40 --maxthread 2 --runs 10
# This command is for evaluating Orbit's compiled MLIR for ResNet-SiLU under base configuration (n=64k,Lm=16,Sw=40).
# The samples are from sample 0 to sample 9
```

#### Checking the correctness of Orbit evaluation

The following script is for checking the correctness of evaluation on MLIRs compiled by Orbit. *Note that the correctness check for micro-benchmark evaluations are not supported.*

**Requirements:**
- True labels: `input_data/<16k|64k>/<model-name>/<activation>/true_labels.txt`
- PyTorch model execution results: `input_data/<16k|64k>/<model-name>/<activation>/plrefs/plref<id>.txt` for `id=0,...,number-1`. *These files are not required for `Orbit-ct-pl` mode*.
- Orbit's HE evaluation results: `mlirs_execute/orbit/<16|64>/<model-name>/<activation>/orbit_*_run<id>.out` for `id=0,...,number-1`. *These files are not required for `PyTorch-Orbit-pl` mode*.
- Orbit's plaintext evaluation results: `mlirs_execute/orbit/<16|64>/<model-name>/<activation>/orbit_*_run<id>_pl.out` for `id=0,...,number-1`. *These files are not required for `PyTorch-Orbit-ct` mode*.

**Script usage:**
```bash
# The current directory should be <orbit-repository>/backend/lattigo/lowering/
python3 check_orbit_one_bench_correctness.py [options]
```

**Options:**
- `--model`, `--act`, `--n`, `--Lm`, `--Sw`: These options are the same as those in the single-sample scripts.
- `--runs <number>`: Check the correctness from sample `0` to sample `number-1`.
- `--mode <mode>`: The checking mode. There are 3 modes:
  - `PyTorch-Orbit-ct`: Compare the accuracy between the plaintext evaluation with model in PyTorch, and the HE evaluation on Orbit's compiled MLIR for current model.
  - `PyTorch-Orbit-pl`: Compare the accuracy between the plaintext evaluation with model in PyTorch, and the plaintext evaluation on Orbit's compiled MLIR for current model.
  - `Orbit-ct-pl`: Compare the accuracy and difference between the HE and plaintext evaluation on Orbit's compiled MLIR for current model.

The standard output will show the accuracy (and the maximum difference between the output vectors).

**Example:**
```bash
python3 check_orbit_one_bench_correctness.py --model ResNet --act SiLU --n 64 --Lm 16 --Sw 40 --runs 10 --mode PyTorch-Orbit-ct
# This command is for checking the correctness of Orbit's compiled MLIR for ResNet-SiLU under base configuration (n=64k,Lm=16,Sw=40). 
# The samples are from sample 0 to sample 9.
# It checks the accuracy of Orbit's MLIR HE execution, and compare it with PyTorch model plaintext evaluation.
```

#### (Optional) Generating Cost Model

We provide a cost model generator to estimate the latency of operations on each level. 
```bash
# The current directory should be <orbit-repository>/backend/lattigo/cost_model/
go run ./cost_model_gen.go [options]
```

**Options:**
- `-n <16|64>`: The number of slots (in 1024s). Default is 64.
- `-btsLb <lb>`: Set the minimum level for bootstrap operation be `lb`. Default is 3.
- `-btsUb <ub>`: Set the maximum level after bootstrap operation be `ub`. Default is 16.

The generated cost model file is `profiled_LATTIGONEW_CPU<16|64>k_<lb>_<ub>.json`.