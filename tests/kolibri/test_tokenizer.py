#!/usr/bin/env python3
"""ds4's Kolibri tokenizer against the HF goldens (tests/golden/kolibri/tokenizer.json).

    python3 test_tokenizer.py --ds4 ./ds4 --model GGUF
"""
import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--ds4", default=os.path.join(HERE, "..", "..", "ds4"))
ap.add_argument("--model", required=True)
args = ap.parse_args()
golden = json.load(open(os.path.join(HERE, "..", "golden", "kolibri", "tokenizer.json")))
cases = [(c["name"], c["text"], c["ids"]) for c in golden["raw"]]
cases += [(c["name"], c["rendered"], c["ids"]) for c in golden["chat"]]
fails = 0
for name, text, want in cases:
    if not text:
        continue
    out = subprocess.run([args.ds4, "-m", args.model, "--dump-tokens", "--raw", "-p", text],
                         capture_output=True, text=True)
    got = json.loads(out.stdout.splitlines()[0]) if out.returncode == 0 and out.stdout else None
    if got != want:
        fails += 1
        first = next((i for i, (a, b) in enumerate(zip(got or [], want)) if a != b), None)
        print(f"FAIL {name}: first diff at {first}\n  want {want[:40]}\n  got  {(got or [])[:40]}\n  {out.stderr[-300:]}")
    else:
        print(f"ok   {name} ({len(want)} tokens)")
print("FAILED" if fails else "ALL OK", f"{fails} of {len(cases)}")
raise SystemExit(1 if fails else 0)
