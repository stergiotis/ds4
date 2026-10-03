#!/usr/bin/env python3
"""Self-consistency of kolibri_ref.py: cached greedy decode must reproduce a
full recompute over prompt + generated tokens, and the sliding mask must
matter once the sequence exceeds the window.

    uv run test_ref_consistency.py --model-dir DIR   (the tiny checkpoint is enough)
"""
import argparse

import numpy as np

from kolibri_ref import Kolibri

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--prompt-len", type=int, default=20)
ap.add_argument("--gen", type=int, default=6)
args = ap.parse_args()

m = Kolibri(args.model_dir)
rng = np.random.default_rng(0)
prompt = [int(x) for x in rng.integers(256, 120000, args.prompt_len)]
gen, prefill_logits, steps = m.greedy(prompt, args.gen)
full = m.logits(m.forward(prompt + gen[:-1]))
cached = np.stack([prefill_logits[-1]] + steps[1:])
tail = full[len(prompt) - 1:]
err = np.abs(tail - cached).max() / np.abs(tail).max()
print(f"window {m.window}, T={len(prompt) + len(gen) - 1}: cached vs full max rel err {err:.2e}")
assert err < 1e-4, err
assert (tail.argmax(-1) == np.array(gen)).all()
# The window must change results beyond it: shrink it and compare.
m.window = 10**9
no_window = m.logits(m.forward(prompt))
d = np.abs(no_window[-1] - prefill_logits[-1]).max()
print(f"removing the window changes last logits by {d:.3e}")
assert d > 1e-3
print("OK")
