#!/usr/bin/env python3
"""Compare ds4's Kolibri forward pass with the numpy goldens.

    uv run compare_golden.py --model GGUF --golden ../golden/kolibri [--backend cpu|rocm] [--only a,b]

For each <name>.npz it runs `ds4 --first-token-test` with the golden prompt
ids (DS4_KOLIBRI_FT_* environment), then reports
  - per-layer residual error at the first and last prompt position
    (max |diff| / max |ref| over the hidden vector),
  - last-position logit error and the top-1 agreement over all positions,
  - whether the greedy continuation matches.
Thresholds are for a Q8_0 GGUF against the fp32 dequantized reference.
"""

import argparse
import glob
import os
import re
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args, g, tmp, backend=None):
    backend = backend or args.backend
    T = len(g["tokens"])
    env = dict(os.environ,
               DS4_KOLIBRI_FT_TOKENS=",".join(map(str, g["tokens"].tolist())),
               DS4_KOLIBRI_FT_GEN=str(len(g["gen"])),
               DS4_KOLIBRI_FT_OUT=os.path.join(tmp, "logits.f32"),
               DS4_KOLIBRI_FT_HIDDEN=os.path.join(tmp, "hidden.f32"),
               DS4_KOLIBRI_FT_EXPERTS=os.path.join(tmp, "experts.i32"))
    if backend == "rocm":
        env["DS4_KOLIBRI_GPU"] = "1"
        if args.chunk:
            env["DS4_KOLIBRI_FT_CHUNK"] = str(args.chunk)
    cmd = [args.ds4, "-m", args.model, "--first-token-test", "--raw", "-p", "x", "--ctx", "4096"]
    cmd += ["--cpu"] if backend == "cpu" else ["--rocm"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"ds4 failed:\n{out.stderr[-2000:]}")
    gen = re.search(r"^greedy:(.*)$", out.stdout, re.M)
    gen = [int(x) for x in gen.group(1).split()] if gen else []
    logits = np.fromfile(env["DS4_KOLIBRI_FT_OUT"], np.float32).reshape(T, -1)
    hidden = np.fromfile(env["DS4_KOLIBRI_FT_HIDDEN"], np.float32)
    hidden = hidden.reshape(T, -1, g["resid_first"].shape[-1])
    experts = None
    if os.path.exists(env["DS4_KOLIBRI_FT_EXPERTS"]):
        experts = np.fromfile(env["DS4_KOLIBRI_FT_EXPERTS"], np.int32).reshape(T, g["experts"].shape[0], -1)
    return logits, hidden, experts, gen


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default=os.path.join(HERE, "..", "..", "ds4"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--backend", choices=("cpu", "rocm"), default="cpu")
    ap.add_argument("--only", default="")
    ap.add_argument("--chunk", type=int, default=0, help="GPU prefill chunk (default: whole prompt)")
    ap.add_argument("--ref", choices=("golden", "cpu"), default="golden",
                    help="cpu: compare against ds4's CPU reference on the same GGUF")
    ap.add_argument("--max-hidden-err", type=float, default=0.05)
    ap.add_argument("--min-top1", type=float, default=0.9)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    failed = 0
    for path in sorted(glob.glob(os.path.join(args.golden, "*.npz"))):
        name = os.path.basename(path)[:-4]
        if args.only and name not in args.only.split(","):
            continue
        g = np.load(path)
        with tempfile.TemporaryDirectory() as tmp:
            logits, hidden, experts, gen = run(args, g, tmp)
        if args.ref == "cpu":
            with tempfile.TemporaryDirectory() as tmp:
                rl, rh, _, rg = run(args, g, tmp, "cpu")
            g = dict(g)
            g.update(resid_first=rh[0], resid_last=rh[-1], last_logits=rl[-1],
                     top_ids=rl.argmax(-1)[:, None], gen=np.array(rg))
        # The GPU graph reports only the final residual (last slot).
        layers = range(hidden.shape[1]) if args.backend == "cpu" else [hidden.shape[1] - 1]
        h_first = [rel(hidden[0, i], g["resid_first"][i]) for i in layers]
        h_last = [rel(hidden[-1, i], g["resid_last"][i]) for i in layers]
        lg = rel(logits[-1], g["last_logits"])
        top1 = float((logits.argmax(-1) == g["top_ids"][:, 0]).mean())
        # Routing agreement: fraction of (position, layer) with the same expert set.
        ref_e = np.sort(g["experts"].transpose(1, 0, 2), -1)
        same = float((np.sort(experts, -1) == ref_e).all(-1).mean()) if experts is not None else float("nan")
        gen_ok = gen == g["gen"].tolist()
        worst = max(max(h_first), max(h_last))
        ok = worst <= args.max_hidden_err and top1 >= args.min_top1 and gen_ok
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: T={len(g['tokens'])} hidden max err {worst:.2e} "
              f"(layer {int(np.argmax(np.maximum(h_first, h_last)))}), last logits {lg:.2e}, "
              f"top1 {top1:.3f}, routing {same:.3f}, greedy {'match' if gen_ok else 'DIFF'}")
        if args.verbose or not ok:
            print("   per-layer last-position err:", " ".join(f"{e:.1e}" for e in h_last))
            if not gen_ok:
                print(f"   greedy ref {g['gen'].tolist()}\n   greedy got {gen}")
    print("ALL OK" if not failed else f"{failed} FAILED")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
