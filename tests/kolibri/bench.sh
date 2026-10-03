#!/bin/sh
# Kolibri 1 benchmark on ROCm: cold prefill of 512, 8192 and 32768 tokens,
# then 128 greedy tokens at that context, three runs each, with peak GPU
# memory (GTT + VRAM, sampled every 0.2 s) and the process's peak RSS.
#
#   tests/kolibri/bench.sh gguf/Kolibri-1-F8.gguf [out.csv]
set -eu
model=$1
out=${2:-/dev/stdout}
cd "$(dirname "$0")/../.."
dev=$(ls -d /sys/class/drm/card*/device | while read d; do [ -f "$d/mem_info_gtt_used" ] && echo "$d"; done | head -1)
gpu_used() { echo $(( $(cat "$dev/mem_info_gtt_used") + $(cat "$dev/mem_info_vram_used") )); }
base=$(gpu_used)
echo "model,ctx,run,prefill_tps,gen_tps,peak_gpu_gib,peak_rss_gib" > "$out"
for ctx in 512 8192 32768; do
  for run in 1 2 3; do
    peakf=$(mktemp)
    echo "$base" > "$peakf"
    ( while :; do u=$(gpu_used); [ "$u" -gt "$(cat "$peakf")" ] && echo "$u" > "$peakf"; sleep 0.2; done ) &
    sampler=$!
    line=$(/usr/bin/time -f "RSS %M" ./ds4-bench --rocm -m "$model" \
             --prompt-file speed-bench/promessi_sposi.txt --ctx-start "$ctx" --ctx-max "$ctx" \
             --gen-tokens 128 2>&1 | grep -E "^$ctx,|^RSS")
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true
    csv=$(echo "$line" | grep "^$ctx,")
    rss=$(echo "$line" | sed -n 's/^RSS //p')
    peak=$(cat "$peakf"); rm -f "$peakf"
    echo "$(basename "$model"),$ctx,$run,$(echo "$csv" | cut -d, -f3),$(echo "$csv" | cut -d, -f5),$(awk "BEGIN{printf \"%.2f\", ($peak-$base)/1073741824}"),$(awk "BEGIN{printf \"%.2f\", $rss/1048576}")" >> "$out"
  done
done
