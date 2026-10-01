#!/usr/bin/env python3
"""Split-pool Q2 tier replay: a Q4 LRU cache of C4 experts and a separate Q2
LRU pool of C2 experts (half the bytes each). A selected expert that is not
Q4-resident and has gate weight < W is served from the Q2 pool (read as Q2 on
a pool miss); every other miss is read as Q4. The heaviest expert of a step
always takes the Q4 path. Memory is held constant: C4 + C2/2 = total.

  tiersim_pool.py [--total N] [--warmup T] TRACE [TRACE ...]
"""
import argparse
import collections
import os
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tw", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiersim_weighted.py"))
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)


class LRU:
    def __init__(self, cap):
        self.cap, self.od = cap, collections.OrderedDict()

    def hit(self, k):
        if k in self.od:
            self.od.move_to_end(k)
            return True
        return False

    def insert(self, k, pinned):
        if self.cap <= 0:
            return
        self.od[k] = None
        while len(self.od) > self.cap:
            for v in self.od:
                if v not in pinned:
                    del self.od[v]
                    break
            else:
                break


def run(steps, c4, c2, W, warmup):
    q4, q2 = LRU(c4), LRU(c2)
    read = uses = q2u = 0
    ws = q2w = 0.0
    for tok, layer, ex, wts in steps:
        pinned = {(layer, e) for e in ex}
        top = max(range(len(ex)), key=lambda i: wts[i])
        for i, (e, w) in enumerate(zip(ex, wts)):
            k = (layer, e)
            r = 0
            at_q2 = False
            if q4.hit(k):
                pass
            elif W > 0 and w < W and i != top:
                at_q2 = True
                if not q2.hit(k):
                    q2.insert(k, pinned)
                    r = 1
            else:
                q4.insert(k, pinned)
                r = 2
            if tok >= warmup:
                read += r
                uses += 1
                ws += w
                if at_q2:
                    q2u += 1
                    q2w += w
    return read, uses, q2u, ws, q2w


ap = argparse.ArgumentParser()
ap.add_argument("--total", type=int, default=5869, help="memory in Q4-expert units")
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("traces", nargs="+")
a = ap.parse_args()
steps = tw.load(a.traces)
ntok = steps[-1][0] + 1 - a.warmup
base = None
for W in (0, 0.20, 0.26):
    for c2 in ((0,) if W == 0 else (0, 1000, 2000, 3000, 4000, 6000)):
        c4 = a.total - c2 // 2
        read, uses, q2u, ws, q2w = run(steps, c4, c2, W, a.warmup)
        pt = read / 2 / ntok
        base = base or pt
        print(f"W {W:.2f}  Q4 {c4:5d}  Q2 pool {c2:5d} ({c2 * 6.75 / 1024:5.1f} GiB)  reads/token {pt:5.1f}"
              f"  vs Q4-only {pt / base - 1:+6.1%}  uses@Q2 {q2u / uses:5.1%}  mass@Q2 {q2w / ws:5.1%}", flush=True)
