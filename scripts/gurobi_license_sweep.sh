#!/usr/bin/env bash
# Sweep every micro/macro benchmark with the size-limited (non-academic) Gurobi
# license and record which ones fail because a partition exceeds the limit.
#
# Classification (per config):
#   PASS         run_orbit.py exited 0 (.mlir produced)
#   LICENSE_FAIL solver hit the size-limited license error (model too large)
#   TIMEOUT      did not finish within PER_TASK_TIMEOUT (slow, NOT a license issue)
#   ERROR(n)     other non-zero exit
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${ORBIT_RESULTS_DIR:-repro_results}/gurobi_sweep"
mkdir -p "$OUT"
RES="$OUT/results.tsv"
: > "$RES"
PER_TASK_TIMEOUT="${PER_TASK_TIMEOUT:-1800}"
TASK_THREADS="${TASK_THREADS:-4}"
PARALLEL="${PARALLEL:-12}"

run() {
    local label="$1"; shift
    local log="$OUT/${label}.log"
    local start end code verdict
    start=$(date +%s)
    timeout "$PER_TASK_TIMEOUT" python3 scripts/optimizer/orbit/run_orbit.py "$@" \
        --threads "$TASK_THREADS" >"$log" 2>&1
    code=$?
    end=$(date +%s)
    if [ "$code" -eq 0 ]; then
        verdict="PASS"
    elif grep -qiE 'too large for size-limited|size-limited license|exceeds the limit' "$log"; then
        verdict="LICENSE_FAIL"
    elif [ "$code" -eq 124 ]; then
        verdict="TIMEOUT"
    else
        verdict="ERROR($code)"
    fi
    printf '%s\t%s\t%ss\n' "$label" "$verdict" "$((end-start))" >> "$RES"
}
export -f run
export OUT RES PER_TASK_TIMEOUT TASK_THREADS ROOT

# Build the full job list: "label -- args..."
jobs_file="$OUT/jobs.txt"
: > "$jobs_file"
emit() { printf '%s\n' "$*" >> "$jobs_file"; }

for m in AlexNet MobileNet SqueezeNet VGG16 ResNet; do for a in SiLU ReLU; do
    emit "base_${m}_${a} -- --model $m --act $a --n 64 --Lm 16 --Sw 40"
done; done
for m in AlexNet SqueezeNet VGG16 ResNet; do for a in SiLU ReLU; do
    emit "16k_${m}_${a} -- --model $m --act $a --n 16 --Lm 16 --Sw 40"
done; done
for m in AlexNet MobileNet SqueezeNet VGG16 ResNet; do for a in SiLU ReLU; do
    emit "lm12_${m}_${a} -- --model $m --act $a --n 64 --Lm 12 --Sw 40"
done; done
for m in AlexNet MobileNet SqueezeNet VGG16 ResNet; do for a in SiLU ReLU; do
    emit "sw51_${m}_${a} -- --model $m --act $a --n 64 --Lm 16 --Sw 51"
done; done
for m in AlexNet MobileNet SqueezeNet VGG16 ResNet; do for a in SiLU ReLU; do
    emit "simvari_${m}_${a} -- --model $m --act $a --n 64 --Lm 16 --Sw 40 --sim-vari"
done; done
emit "micro-comppart_comp_part -- --model CompPart --act SiLU --n 64 --Lm 16 --Sw 40"
emit "micro-comppart_comp_nopart -- --model CompPart --act SiLU --n 64 --Lm 16 --Sw 40 --nopart"
emit "micro-comppart_nocomp_part -- --model CompPart --act SiLU --n 64 --Lm 16 --Sw 40 --nocomp"
emit "micro-comppart_nocomp_nopart -- --model CompPart --act SiLU --n 64 --Lm 16 --Sw 40 --nocomp --nopart"
emit "micro-bypass_ResNet_ReLU_64 -- --model ResNet --act ReLU --n 64 --Lm 16 --Sw 40 --nobypass"
emit "micro-bypass_ResNet_SiLU_64 -- --model ResNet --act SiLU --n 64 --Lm 16 --Sw 40 --nobypass"
emit "micro-bypass_ResNet_SiLU_16 -- --model ResNet --act SiLU --n 16 --Lm 16 --Sw 40 --nobypass"
emit "micro-bypass_ResNet_ReLU_16 -- --model ResNet --act ReLU --n 16 --Lm 16 --Sw 40 --nobypass"
for m in AlexNet SqueezeNet VGG16 ResNet; do for a in SiLU ReLU; do
    emit "micro-reqbp_${m}_${a} -- --model $m --act $a --n 16 --Lm 16 --Sw 40 --qbp"
done; done

echo "Total jobs: $(wc -l < "$jobs_file"), parallelism: $PARALLEL, threads/task: $TASK_THREADS, timeout: ${PER_TASK_TIMEOUT}s"

# Run in parallel. Each line: "label -- args"
cat "$jobs_file" | xargs -P "$PARALLEL" -I {} bash -c '
    line="{}"
    label="${line%% -- *}"
    args="${line#* -- }"
    run "$label" $args
'

echo "=== DONE. $(wc -l < "$RES") results in $RES ==="
sort "$RES"
