#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"
THREADS="${ORBIT_THREADS:-2}"
RESULT_ROOT="${ORBIT_RESULTS_DIR:-repro_results}"
RUN_ID="${ORBIT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="$RESULT_ROOT/$RUN_ID"
SUMMARY="$OUT_DIR/summary.txt"

mkdir -p "$OUT_DIR"

usage() {
    cat <<'EOF'
Usage:
  scripts/reproduce.sh [mode]

Modes:
  quick                  Run pytest and a small optimizer benchmark smoke test.
  tests                  Run pytest only.
  benchmark-smoke        Compile mlirs_input/motivation.mlir with the toy cost model.
  deps                   Install Python test/runtime dependencies with pip.
  compile-base           Run the README base compilation suite.
  compile-16k            Run the README 16k compilation suite.
  compile-lm12           Run the README Lm=12 compilation suite.
  compile-sw51           Run the README Sw=51 compilation suite.
  compile-sim-vari       Run the simulated-variadic compilation suite.
  compile-micro-comppart Run the compression/partitioning micro suite.
  compile-micro-bypass   Run the bypass micro suite.
  compile-micro-reqbp    Run the qbp reuse micro suite.
  compile-all            Run all compilation suites above.
  execute                Evaluate a compiled MLIR through the Lattigo backend.

Execute usage:
  scripts/reproduce.sh execute --model M --act A --n N --Lm L --Sw S \
                               [--run ID] [--cmt CMT] [--Csw CSW] [--plain]

  Requires the Lattigo backend (scripts/setup_dependencies.sh backend) and the
  frontend-generated input_data/ and input_constants/ files (not shipped; mount
  them). The MLIR is compiled on demand for the default configuration if absent.

Orbit solves its ILPs with Gurobi (gurobipy). The image ships Gurobi's bundled
size-limited license, which is enough for every partitioned benchmark; only the
--nopart configurations need a full/academic Gurobi license.

Environment:
  ORBIT_THREADS          Solver threads. Default: 2.
  ORBIT_RESULTS_DIR      Result log root. Default: repro_results.
  ORBIT_RUN_ID           Result subdirectory name. Default: UTC timestamp.
EOF
}

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

record() {
    printf '%s\n' "$*" | tee -a "$SUMMARY"
}

run_logged() {
    local name="$1"
    shift
    local log="$OUT_DIR/${name}.log"

    record "[$(timestamp)] RUN $*"
    if "$@" >"$log" 2>&1; then
        record "[$(timestamp)] PASS $name"
    else
        local status=$?
        record "[$(timestamp)] FAIL $name (exit $status)"
        record "Last 40 lines from $log:"
        tail -40 "$log" | tee -a "$SUMMARY"
        return "$status"
    fi
}

summarize_optimizer_log() {
    local log="$1"
    if [[ ! -f "$log" ]]; then
        record "Missing optimizer log: $log"
        return 1
    fi

    record "Optimizer summary from $log:"
    grep -E 'Built original DAG|After Compression|Final assignment latency|Final tdag latency|Orbit Compilation time' "$log" \
        | tee -a "$SUMMARY" || record "No optimizer summary lines found in $log"
}

run_tests() {
    run_logged pytest python3 -m pytest -q
    record "Pytest summary:"
    tail -5 "$OUT_DIR/pytest.log" | tee -a "$SUMMARY"
}

run_benchmark_smoke() {
    local output="mlirs_output/repro/motivation.mlir"
    run_logged benchmark-smoke \
        python3 -u -m scripts.optimizer.orbit.optimizer \
            --inputfile mlirs_input/motivation.mlir \
            --outputfile "$output" \
            --costjson cost_models/toy_backend.json \
            --waterscale 40 \
            --threads "$THREADS"
    summarize_optimizer_log "$OUT_DIR/benchmark-smoke.log"
}

run_orbit_bench() {
    local model="$1"
    local act="$2"
    local n="$3"
    local Lm="$4"
    local Sw="$5"
    shift 5
    local flags=("$@")

    local bypass="bypass"
    local qbp="noqbp"
    local comp="comp"
    local part="part"
    local simvari=""
    local flag
    for flag in "${flags[@]}"; do
        case "$flag" in
            --nobypass) bypass="nobypass" ;;
            --qbp) qbp="qbp" ;;
            --nocomp) comp="nocomp" ;;
            --nopart) part="nopart" ;;
            --sim-vari) simvari="_simvari" ;;
        esac
    done

    local this_n="$n"
    if [[ "$Lm" != "16" ]]; then
        this_n="$Lm"
    fi

    local outname="orbit_${model}${act}${n}k_Lm${Lm}_Sw${Sw}_${bypass}_${qbp}_${comp}_${part}${simvari}"
    local orbit_log="mlirs_output/orbit/${model}/${Sw}/${act}/${this_n}/${outname}.txt"
    local orbit_err="${orbit_log%.txt}.err"

    run_logged "compile-${outname}" \
        python3 scripts/optimizer/orbit/run_orbit.py \
            --model "$model" \
            --act "$act" \
            --n "$n" \
            --Lm "$Lm" \
            --Sw "$Sw" \
            --threads "$THREADS" \
            "${flags[@]}"

    summarize_optimizer_log "$orbit_log"
    if [[ -s "$orbit_err" ]]; then
        record "Non-empty stderr log: $orbit_err"
        tail -40 "$orbit_err" | tee -a "$SUMMARY"
        return 1
    fi
}

run_compile_suite() {
    local suite="$1"
    local model
    local act
    local no_comp
    local no_part

    case "$suite" in
        base)
            for model in AlexNet MobileNet SqueezeNet VGG16 ResNet; do
                for act in SiLU ReLU; do
                    run_orbit_bench "$model" "$act" 64 16 40
                done
            done
            ;;
        16k)
            for model in AlexNet SqueezeNet VGG16 ResNet; do
                for act in SiLU ReLU; do
                    run_orbit_bench "$model" "$act" 16 16 40
                done
            done
            ;;
        lm12)
            for model in AlexNet MobileNet SqueezeNet VGG16 ResNet; do
                for act in SiLU ReLU; do
                    run_orbit_bench "$model" "$act" 64 12 40
                done
            done
            ;;
        sw51)
            for model in AlexNet MobileNet SqueezeNet VGG16 ResNet; do
                for act in SiLU ReLU; do
                    run_orbit_bench "$model" "$act" 64 16 51
                done
            done
            ;;
        sim-vari)
            for model in AlexNet MobileNet SqueezeNet VGG16 ResNet; do
                for act in SiLU ReLU; do
                    run_orbit_bench "$model" "$act" 64 16 40 --sim-vari
                done
            done
            ;;
        micro-comppart)
            for no_comp in "" "--nocomp"; do
                for no_part in "" "--nopart"; do
                    run_orbit_bench CompPart SiLU 64 16 40 ${no_comp:+"$no_comp"} ${no_part:+"$no_part"}
                done
            done
            ;;
        micro-bypass)
            for act in ReLU SiLU; do
                run_orbit_bench ResNet "$act" 64 16 40 --nobypass
            done
            for act in SiLU ReLU; do
                run_orbit_bench ResNet "$act" 16 16 40 --nobypass
            done
            ;;
        micro-reqbp)
            for model in AlexNet SqueezeNet VGG16 ResNet; do
                for act in SiLU ReLU; do
                    run_orbit_bench "$model" "$act" 16 16 40 --qbp
                done
            done
            ;;
        *)
            record "Unknown compilation suite: $suite"
            return 2
            ;;
    esac
}

run_execute() {
    # Evaluate a compiled Orbit MLIR through the Lattigo backend.
    local model="" act="" n="" Lm="" Sw="" run="0" cmt="bypass_noqbp_comp_part" Csw="" plain=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model) model="$2"; shift 2 ;;
            --act)   act="$2";   shift 2 ;;
            --n)     n="$2";     shift 2 ;;
            --Lm)    Lm="$2";    shift 2 ;;
            --Sw)    Sw="$2";    shift 2 ;;
            --run)   run="$2";   shift 2 ;;
            --cmt)   cmt="$2";   shift 2 ;;
            --Csw)   Csw="$2";   shift 2 ;;
            --plain) plain="--plain"; shift ;;
            *) record "Unknown execute option: $1"; return 2 ;;
        esac
    done

    if [[ -z "$model" || -z "$act" || -z "$n" || -z "$Lm" || -z "$Sw" ]]; then
        record "execute requires --model --act --n --Lm --Sw (and optional --run --cmt --Csw --plain)"
        return 2
    fi

    # Backend must be built first (kept out of the image; see setup_dependencies.sh).
    local lowering="backend/lattigo/lowering"
    if [[ ! -f "$lowering/run_orbit_one_eval.py" ]]; then
        record "Lattigo backend not found at $lowering."
        record "Build it first (network required), e.g. inside the container with a persisted volume:"
        record "  ./scripts/setup_dependencies.sh backend"
        return 1
    fi

    local benchmark="${model}${act}${n}k"
    local this_n="$n"
    if [[ "$Lm" != "16" ]]; then
        this_n="$Lm"
    fi

    # Compile the MLIR on demand for the default configuration if it is missing.
    local outname="orbit_${benchmark}_Lm${Lm}_Sw${Sw}"
    [[ -n "$Csw" ]] && outname="${outname}_Csw${Csw}"
    outname="${outname}_${cmt}"
    local mlir="mlirs_output/orbit/${model}/${Sw}/${act}/${this_n}/${outname}.mlir"
    if [[ ! -f "$mlir" ]]; then
        if [[ "$cmt" == "bypass_noqbp_comp_part" ]]; then
            record "Compiled MLIR missing; compiling it first: $mlir"
            run_orbit_bench "$model" "$act" "$n" "$Lm" "$Sw" ${Csw:+--Csw "$Csw"}
        else
            record "Compiled MLIR not found for cmt='$cmt': $mlir"
            record "Compile it first with the matching scripts/reproduce.sh compile-* mode."
            return 1
        fi
    fi

    # Frontend-generated data is required and is not shipped in the image.
    local data_model="$model" data_act_dir
    if [[ "$model" == "CompPart" ]]; then
        data_model="ResNet"   # CompPart reuses ResNet inputs/constants
    fi
    local cst="input_constants/${data_model}${act}${n}k_hecate.cst"
    local inp="input_data/${n}k/${data_model,,}/${act,,}/inputs/input${run}.txt"
    if [[ ! -f "$cst" || ! -f "$inp" ]]; then
        record "Execution data not found:"
        [[ -f "$cst" ]] || record "  missing constants: $cst"
        [[ -f "$inp" ]] || record "  missing input sample: $inp"
        record "These are generated by the frontend (GBs) and must be provided/mounted."
        record "See backend/README.md and frontend/README.md."
        return 1
    fi

    run_logged "execute-${outname}_run${run}${plain:+_pl}" \
        bash -c 'cd "$1" && shift && python3 run_orbit_one_eval.py "$@"' _ "$lowering" \
            --model "$model" --act "$act" --n "$n" --Lm "$Lm" --Sw "$Sw" \
            --run "$run" --cmt "$cmt" ${Csw:+--Csw "$Csw"} ${plain:+$plain}

    local exec_log="mlirs_execute/orbit/${n}/${model}/${act}/${outname}_run${run}${plain:+_pl}.log"
    if [[ -f "$exec_log" ]]; then
        record "Execution log: $exec_log"
        tail -20 "$exec_log" | tee -a "$SUMMARY"
    fi
}

record "Orbit reproducibility run"
record "Mode: $MODE"
record "Solver: gurobi"
record "Threads: $THREADS"
record "Results: $OUT_DIR"

case "$MODE" in
    quick)
        run_tests
        run_benchmark_smoke
        ;;
    tests)
        run_tests
        ;;
    benchmark-smoke)
        run_benchmark_smoke
        ;;
    deps)
        run_logged install-python-deps python3 -m pip install -r requirements-dev.txt
        ;;
    compile-base)
        run_compile_suite base
        ;;
    compile-16k)
        run_compile_suite 16k
        ;;
    compile-lm12)
        run_compile_suite lm12
        ;;
    compile-sw51)
        run_compile_suite sw51
        ;;
    compile-sim-vari)
        run_compile_suite sim-vari
        ;;
    compile-micro-comppart)
        run_compile_suite micro-comppart
        ;;
    compile-micro-bypass)
        run_compile_suite micro-bypass
        ;;
    compile-micro-reqbp)
        run_compile_suite micro-reqbp
        ;;
    compile-all)
        run_compile_suite base
        run_compile_suite 16k
        run_compile_suite lm12
        run_compile_suite sw51
        run_compile_suite sim-vari
        run_compile_suite micro-comppart
        run_compile_suite micro-bypass
        run_compile_suite micro-reqbp
        ;;
    execute)
        run_execute "${@:2}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac

record "[$(timestamp)] DONE"
record "Summary written to $SUMMARY"
