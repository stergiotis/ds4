#!/bin/bash
# Reproducible ds4 benchmark for Strix Halo (gfx1151, ROCm, SSD streaming).
#
# Runs ds4-bench with teacher-forced decode on a fixed prompt, so every run
# routes the same tokens through the same experts and timings are comparable.
# Frontier logits are dumped for an exact before/after check with
# gguf-tools/quality-testing/compare_frontier_logits.py (see strix/compare.sh).
#
#   strix/bench.sh MODEL.gguf LABEL [extra ds4-bench args...]
#
# DS4_* environment variables pass through to ds4 and are recorded, e.g.
#   DS4_GLM_MEMORY_GUARD_RESERVE_GB=12 strix/bench.sh $Q4 q4-reserve12
#
# Environment:
#   DS4_BENCH_OUT   results root (default ~/.local/share/ds4/bench)
#   DS4_BENCH_CTX   last frontier (default 8192); first is 2048, step 2048
#   DS4_BENCH_GEN   decode tokens per frontier (default 64)
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
model=${1:?usage: strix/bench.sh MODEL.gguf LABEL [ds4-bench args...]}
label=${2:?usage: strix/bench.sh MODEL.gguf LABEL [ds4-bench args...]}
shift 2

root=${DS4_BENCH_OUT:-$HOME/.local/share/ds4/bench}
out="$root/$(date +%Y%m%d-%H%M%S)-$label"
mkdir -p "$out/logits"

{
    echo "label: $label"
    echo "model: $model"
    echo "commit: $(git -C "$here" describe --always --dirty)"
    echo "branch: $(git -C "$here" rev-parse --abbrev-ref HEAD)"
    echo "date: $(date -Is)"
    echo "kernel: $(uname -r)"
    echo "gtt_total_gib: $(( $(cat /sys/class/drm/card*/device/mem_info_gtt_total | head -1) >> 30 ))"
    echo "args: $*"
    echo "env:"
    env | grep -E '^(DS4_|HIP_|AMD_|ROC)' | sort | sed 's/^/  /' || true
} > "$out/meta.txt"

# Disk throughput (MB/s read on the NVMe) and GPU-mapped memory (GTT, MiB)
# while the benchmark runs. The GTT peak shows how close a configuration comes
# to the kernel's ttm.pages_limit, which ds4's memory plan does not fully model.
iostat -d -m nvme0n1 5 > "$out/iostat.txt" 2>&1 &
iostat_pid=$!
gtt_file=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
( while :; do echo "$(date +%T) $(( $(cat "$gtt_file") >> 20 ))"; sleep 1; done ) > "$out/gtt.txt" &
gtt_pid=$!
trap 'kill $iostat_pid $gtt_pid 2>/dev/null || true' EXIT

"$here/ds4-bench" --rocm --ssd-streaming -m "$model" \
    --prompt-file "$here/speed-bench/promessi_sposi.txt" \
    --ctx-start 2048 --ctx-max "${DS4_BENCH_CTX:-8192}" --step-incr 2048 \
    --gen-tokens "${DS4_BENCH_GEN:-64}" --teacher-forced-decode \
    --csv "$out/result.csv" --dump-frontier-logits-dir "$out/logits" \
    "$@" 2> "$out/stderr.log"

grep -E 'cache target|memory:|stream read' "$out/stderr.log" > "$out/summary.txt" || true
kill "$gtt_pid" 2>/dev/null || true
sort -k2 -n "$out/gtt.txt" | tail -1 |
    awk -v total="$(( $(cat /sys/class/drm/card*/device/mem_info_gtt_total | head -1) >> 20 ))" \
        '{printf "gtt peak: %.2f GiB of %.2f GiB\n", $2 / 1024, total / 1024}' >> "$out/summary.txt"
echo "--- result.csv" >> "$out/summary.txt"
cat "$out/result.csv" >> "$out/summary.txt"
cat "$out/summary.txt"
echo "results: $out"
