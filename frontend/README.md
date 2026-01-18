## Frontend Instructions

Orbit extends Dacapo's frontend interface, which can generate FHE programs represented in MLIR's FHE dialect based on Python scripts.

### Install Dacapo

This is the link of [Dacapo](https://github.com/corelab-src/dacapo). Please first clone the repository, and then install Orbit's patch on Dacapo:

```bash
# The current directory should be <orbit-repository>/frontend/
git clone https://github.com/corelab-src/dacapo.git
./patch_dacapo.sh
```

#### Requirements 
```
Ninja   
git  
cmake >= 3.22.1  
python >= 3.10  
clang,clang++ >= 14.0.0  
```

#### Install MLIR 
```bash
# The current directory for installing MLIR is unrestricted
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-18.1.2
cmake -GNinja -Bbuild \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  llvm
cmake --build build
sudo cmake --install build
cd .. 
```

#### Install SEAL 
```bash
# The current directory for installing SEAL is unrestricted
git clone https://github.com/microsoft/SEAL.git
cd SEAL
git checkout 4.0.0
cmake -S . -B build
cmake --build build
sudo cmake --install build
cd .. 
```

#### Build Hecate 
```bash
# The current directory should be <orbit-repository>/frontend/
cd dacapo
cmake -S . -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build 
```

#### Configure Hecate 
```bash
# The current directory should be <orbit-repository>/frontend/dacapo/
python3 -m venv .venv
source .venv/bin/activate
source config.sh 
```

#### Install Hecate Python Binding 
```bash
# The current directory should be <orbit-repository>/frontend/dacapo/
pip install -r requirements.txt
./install.sh
```

### Usage

We provide several scripts for reproducing Orbit's evaluation results. For detailed usage, please check Dacapo's original tutorial.

**If you only want to reproduce Orbit's compilation results, the following scripts are not necessary.**

#### Benchmark MLIR + Plaintext Generation

The following script can trace all benchmarks' Python file to MLIR files (`<orbit-repository>/mlirs_input/<benchmark>.mlir`) and corresponding plaintext files (`<orbit-repository>/input_constants/<benchmark>_hecate.cst`).

```bash
# The current directory should be <orbit-repository>/frontend/dacapo/
../dacapo_patch/gen_all_mlirs.sh
```

#### Benchmark Samples Generation

The following script can generate samples for benchmark evaluation from CIFAR-10 dataset.

```bash
# The current directory should be <orbit-repository>/frontend/dacapo/
python3 examples/tests/gen_input_data.py <sample-num:integer>
# e.g. The following script generates 10 samples for each benchmark. 
# python3 examples/tests/gen_input_data.py 10 
```

The samples can be found at `<orbit-repository>/input_data/<16k|64k>/<model-name>/<relu|silu>/`. For the id-th sample, the input sample file is `inputs/input<id>.txt`, the Pytorch model evaluation result on this sample is `plrefs/plref<id>.txt`, and the true label of this sample can be found in `true_labels.txt`.

### Dacapo's Original Tutorial (Partial)

#### Trace one example python file to MLIR

```bash
hc-trace <example-name>
```

e.g.

```bash
hc-trace ResNetSiLU64k
```

#### Compile the traced MLIR and Check the optimized code 
```bash
hbt <pars|dacapo> <waterline:integer> <example-name> <library-name> <hardware>
```
e.g., 
```bash
hbt dacapo 40 ResNetSiLU64k HEAAN GPU
```
This command will print like this:
```
Estimated Latency: 13.600457 (sec)
Number of Bootstrapping: 19
===---------------------------====
  ... Execution Time Report ....
```
You can see the optimized code in "$hecate-compiler/examples/optimized/dacapo/ResNetSiLU64k.40.earth.mlir"

If you see an error message like "error: 'earth.bootstrap' op failed to infer returned types",\
just wait as it is in the normal compilation process.

