#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-python}"
BUILD_JOBS="${ORBIT_BUILD_JOBS:-$(nproc)}"
DEPS_DIR="${ORBIT_DEPS_DIR:-$ROOT/.deps}"

usage() {
    cat <<'EOF'
Usage:
  scripts/setup_dependencies.sh [python|backend|frontend|all]

Targets:
  python    Install Python runtime/test dependencies from requirements-dev.txt.
  backend   Clone Lattigo, apply Orbit's patch, run go mod tidy, and build fhe_binary.
  frontend  Build LLVM/MLIR, SEAL, and patched Dacapo/Hecate from source.
  all       Run python, backend, and frontend.

Notes:
  frontend is large. It builds LLVM/MLIR from source and may take a long time.
  ORBIT_DEPS_DIR controls where external source trees are cloned. Default: .deps.
EOF
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        return 1
    fi
}

install_cmake_build() {
    if [[ "$(id -u)" == "0" ]]; then
        cmake --install build
    else
        need_cmd sudo
        sudo cmake --install build
    fi
}

setup_python() {
    python3 -m pip install -r requirements-dev.txt
}

setup_backend() {
    need_cmd git
    need_cmd go

    mkdir -p backend
    if [[ ! -d backend/lattigo/.git ]]; then
        git clone https://github.com/tuneinsight/lattigo.git backend/lattigo
    fi

    (
        cd backend
        ./patch_lattigo.sh
    )

    (
        cd backend/lattigo
        go mod tidy
        cd lowering
        go build -o fhe_binary ./fhe
    )
}

setup_frontend() {
    need_cmd clang
    need_cmd clang++
    need_cmd cmake
    need_cmd git
    need_cmd ninja
    need_cmd python3

    mkdir -p "$DEPS_DIR"

    if [[ ! -d "$DEPS_DIR/llvm-project/.git" ]]; then
        git clone https://github.com/llvm/llvm-project.git "$DEPS_DIR/llvm-project"
    fi
    (
        cd "$DEPS_DIR/llvm-project"
        git checkout llvmorg-18.1.2
        cmake -GNinja -Bbuild \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_PROJECTS=mlir \
            -DLLVM_INSTALL_UTILS=ON \
            -DLLVM_TARGETS_TO_BUILD=host \
            llvm
        cmake --build build --parallel "$BUILD_JOBS"
        install_cmake_build
    )

    if [[ ! -d "$DEPS_DIR/SEAL/.git" ]]; then
        git clone https://github.com/microsoft/SEAL.git "$DEPS_DIR/SEAL"
    fi
    (
        cd "$DEPS_DIR/SEAL"
        git checkout 4.0.0
        cmake -S . -B build
        cmake --build build --parallel "$BUILD_JOBS"
        install_cmake_build
    )

    if [[ ! -d frontend/dacapo/.git ]]; then
        git clone https://github.com/corelab-src/dacapo.git frontend/dacapo
    fi

    (
        cd frontend
        ./patch_dacapo.sh
    )

    (
        cd frontend/dacapo
        cmake -S . -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
        cmake --build build --parallel "$BUILD_JOBS"
        python3 -m venv .venv
        . .venv/bin/activate
        . config.sh
        python3 -m pip install -r requirements.txt
        ./install.sh
    )
}

case "$MODE" in
    python)
        setup_python
        ;;
    backend)
        setup_backend
        ;;
    frontend)
        setup_frontend
        ;;
    all)
        setup_python
        setup_backend
        setup_frontend
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
