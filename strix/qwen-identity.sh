#!/bin/bash
# Greedy MTP must reproduce plain decode byte for byte on Qwen3.8.
#
#   strix/qwen-identity.sh MODEL.gguf [TOKENS] [OUT_DIR]
#
# For every prompt in strix/qwen-prompts/, decodes TOKENS greedy tokens
# (default 384, thinking off) without MTP and with MTP at the automatic depth
# and at fixed depths 2 and 3, and compares the outputs. Exits 0 only when
# every MTP run matches its plain run. A mismatch means a kernel rounds a
# verify row differently from plain decode; DS4_QWEN4_ROW_HASH=1 finds the
# first logits row that differs and DS4_QWEN4_OP_HASH_POS=<pos> the first op.
# Needs the GPU to itself; about 2 minutes per prompt.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
model=${1:?usage: strix/qwen-identity.sh MODEL.gguf [TOKENS] [OUT_DIR]}
tokens=${2:-384}
out=${3:-$(mktemp -d "${TMPDIR:-/tmp}/qwen-identity.XXXXXX")}
mkdir -p "$out"
run() { # name prompt depth(plain|auto|2|3)
    local flags="" env=DS4_NOP=1
    [ "$3" = plain ] || flags="--mtp-timing"
    case $3 in 2|3) env="DS4_QWEN4_MTP_DEPTH=$3";; esac
    env $env "$here/ds4" --rocm -m "$model" --ctx 8192 -n "$tokens" --nothink --temp 0 $flags \
        --prompt-file "$2" > "$out/$1.out" 2> "$out/$1.err"
}
fail=0
for prompt in "$here"/strix/qwen-prompts/*.txt; do
    p=$(basename "$prompt" .txt)
    if ! run "$p-plain" "$prompt" plain; then
        echo "$p: plain run failed (see $out/$p-plain.err)"; fail=1; continue
    fi
    line="$p:"
    for d in auto 2 3; do
        if ! run "$p-$d" "$prompt" "$d"; then line="$line depth $d FAILED;"; fail=1; continue; fi
        if cmp -s "$out/$p-plain.out" "$out/$p-$d.out"; then
            line="$line depth $d same;"
        else
            at=$(cmp "$out/$p-plain.out" "$out/$p-$d.out" | awk '{print $5}' | tr -d ,)
            line="$line depth $d DIFFERS at byte $at;"; fail=1
        fi
    done
    echo "$line"
done
[ $fail = 0 ] && echo "identity: all MTP runs match plain decode ($out)" || echo "identity: FAILED ($out)"
exit $fail
