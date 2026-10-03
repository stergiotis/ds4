#!/usr/bin/env python3
"""How far do numerical variants of the reference move from the golden?

    uv run sensitivity.py --model-dir SNAPSHOT --golden ../golden/kolibri/de_story.npz \
        --variant act_quant|weight_q8 [--out FILE.npz]

act_quant:  vLLM's dynamic FP8 activation quantization (how the reference
            actually runs its FP8 checkpoint).
weight_q8:  weights rounded through ggml Q8_0, i.e. what the Q8_0 GGUF holds;
            ds4's CPU reference on that GGUF should reproduce this closely.
Prints the per-layer residual error at the last position, the routing
agreement and the greedy continuation, like compare_golden.py.
"""
import argparse

import numpy as np

from kolibri_ref import Kolibri

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--golden", required=True)
ap.add_argument("--variant", choices=("act_quant", "weight_q8"), required=True)
ap.add_argument("--out", default="")
args = ap.parse_args()
g = np.load(args.golden)
m = Kolibri(args.model_dir, act_quant=args.variant == "act_quant", weight_q8=args.variant == "weight_q8")
trace = {}
gen, lg, steps = m.greedy(g["tokens"].tolist(), len(g["gen"]), trace)
resid = np.stack(trace["resid"])
experts = np.stack(trace["experts"])
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
errs = [rel(resid[i, -1], g["resid_last"][i]) for i in range(resid.shape[0])]
same = float((np.sort(experts, -1) == np.sort(g["experts"], -1)).all(-1).mean())
print(f"{args.variant}: hidden err by layer (last position):", " ".join(f"{e:.1e}" for e in errs))
print(f"{args.variant}: last logits err {rel(lg[-1], g['last_logits']):.2e}, routing {same:.3f}, "
      f"top1 {(lg.argmax(-1) == g['top_ids'][:, 0]).mean():.3f}")
print(f"{args.variant}: greedy {gen}\n    golden {g['gen'].tolist()}")
if args.out:
    np.savez_compressed(args.out, resid_last=resid[:, -1], resid_first=resid[:, 0], last_logits=lg[-1],
                        gen=np.array(gen), experts=experts.astype(np.int16), tokens=g["tokens"],
                        top_ids=np.argsort(-lg, -1)[:, :32].astype(np.int32))
