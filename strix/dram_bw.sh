#!/bin/bash
# DRAM read bandwidth per LPDDR5X channel on Strix Halo, from the AMD data
# fabric PMU (the UMC PMUs expose 0 counters on this APU).
#
#   strix/dram_bw.sh [SECONDS]
#
# Needs: sudo modprobe amd_uncore; sudo sysctl kernel.perf_event_paranoid=-1
# Channel N's read-beat event is event=0x1f+0x40*N, umask=0xffe (perf names
# channels 0-11; this APU has 16). A beat is 64 bytes, which matches first-
# principles estimates for dense-model decode (~203 GB/s at 16.7 tok/s for a
# 28.6 GB Q8 model). The fabric has 8 counters, so channels are read in two
# groups of 8.
set -euo pipefail
secs=${1:-5}
[[ -d /sys/bus/event_source/devices/amd_df ]] || { echo "amd_df PMU missing: sudo modprobe amd_uncore" >&2; exit 1; }
total=0
for grp in "0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15"; do
    ev=()
    for n in $grp; do
        ev+=(-e "amd_df/event=$(printf '0x%x' $((0x1f + 0x40 * n))),umask=0xffe,name=rd$n/")
    done
    while IFS=, read -r count _ name _; do
        [[ $name == rd* ]] || continue
        gbs=$(awk -v c="$count" -v s="$secs" 'BEGIN{printf "%.2f", c * 64 / s / 1e9}')
        printf "%-5s %7s GB/s\n" "$name" "$gbs"
        total=$(awk -v t="$total" -v g="$gbs" 'BEGIN{print t + g}')
    done < <(perf stat -a -x, "${ev[@]}" -- sleep "$secs" 2>&1)
done
printf "total %7.1f GB/s read (LPDDR5X-8000 x 256 bit peak: 256 GB/s)\n" "$total"
