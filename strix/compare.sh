#!/bin/bash
# Compare the frontier logits of two strix/bench.sh runs.
#
#   strix/compare.sh BASELINE_RUN_DIR CANDIDATE_RUN_DIR
#
# Exits 0 only when every frontier is float32-identical (the bar for pure I/O,
# caching and scheduling changes). The JSON report in the candidate's run
# directory also carries max-abs, RMSE and top-1 agreement for changes that
# alter arithmetic order, which need a looser, stated envelope.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
base=${1:?usage: strix/compare.sh BASELINE_RUN_DIR CANDIDATE_RUN_DIR}
cand=${2:?usage: strix/compare.sh BASELINE_RUN_DIR CANDIDATE_RUN_DIR}

read -r frontiers ctx model backend quality bits vocab < <(python3 - "$base/logits" <<'EOF'
import json, pathlib, sys
files = sorted(pathlib.Path(sys.argv[1]).glob("frontier_*.logits.json"))
if not files:
    sys.exit(f"no frontier dumps in {sys.argv[1]}")
fronts = []
for f in files:
    fronts.append(int(f.name.split("_")[1].split(".")[0]))
d = json.loads(files[0].read_text())
print(",".join(map(str, fronts)), d["ctx"], d["model"], d["backend"],
      "true" if d["quality"] else "false", d["quant_bits"], d["vocab"])
EOF
)

report="$cand/compare-vs-$(basename "$base").json"
rm -f "$report"
python3 "$here/gguf-tools/quality-testing/compare_frontier_logits.py" \
    "$base/logits" "$cand/logits" \
    --frontiers ${frontiers//,/ } --ctx "$ctx" --model "$model" \
    --backend "$backend" --quality "$quality" --quant-bits "$bits" \
    --vocab "$vocab" --output "$report"
