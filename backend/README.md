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

#### Evaluate Orbit on One Benchmark, One sample

Please make sure that the corresponding sample input data and plaintext files are generated. If not, please use scripts in frontend to generate.

Requirements:
- Sample input file: `input_data/<16k|64k>/<model-name>/<activation>/inputs/input<id>.txt`
- MLIR plaintext file: `input_constants/<model-name><activation><16k|64k>_hecate.cst`
- Orbit's compiled MLIR file: `mlirs_output/orbit/<model-name>/<waterline-scale>/<activation>/<16|64>/orbit_*.mlir`

Script usage:
```bash
# The current directory should be <orbit-repository>/backend/lattigo/lowering/
python run_orbit_one_eval.py [options]
```

Options:
- `--model <model-name>`: (Required) The model name.
- `--act <activation>`: (Required) The activation function name.
- `--n <16|64>`: (Required) The number of slots (in 1024s). 
- `--Lm <maxlevle>`: (Required) The number of effective levels for computation (i.e. $L_{bts}$ in the paper).
- `--Sw <waterline>`: (Required) The minimum scale attribute (i.e. $\log S_{min}$ in the paper).
- `--run <id>`: (Required) The sample index.
- `--plain`: If enabled, execute the MLIR file in plaintext mode.
- `--cmt <orbit-comment>`: Specify the comment in the name of Orbit's compiled MLIR file. The default comment is for macro-benchmarks evaluation. Check `run_orbit_one_eval.py` for comments in micro-benchmarks evaluation.
- `--lcmt <log-comment>`: Additional comments appended to the log file.
- `--Csw`: Specify the minimum scale attribute for plaintexts. This experimental option is not used in Orbit's evaluation.


