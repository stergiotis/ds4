#!/usr/bin/env python3
"""LRU with a pinned hotlist: the H most frequent experts of a training trace
stay resident; the rest of the capacity is LRU. Evaluated on a test trace
(or the second half of one trace) against plain LRU at the same capacity."""
import argparse, collections, importlib.util, os
spec = importlib.util.spec_from_file_location("cs", os.path.join(os.path.dirname(__file__), "cachesim.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)

class PinnedLRU(cs.LRU):
    def __init__(self, cap, hot):
        super().__init__(cap - len(hot))
        self.hot = hot
    def hit(self, k):
        return k in self.hot or super().hit(k)

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=5869)
ap.add_argument("--train", nargs="+", required=True)
ap.add_argument("--test", nargs="+", required=True)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--split", action="store_true", help="train on first half of --train, test on its second half")
a = ap.parse_args()
train = cs.load(a.train)
if a.split:
    mid = train[-1][0] // 2
    test = [(t - mid, l, e) for t, l, e in train if t >= mid]
    train = [s for s in train if s[0] < mid]
else:
    test = cs.load(a.test)
freq = collections.Counter((l, e) for _, l, ex in train for e in ex)
n = sum(len(s[2]) for s in test if s[0] >= a.warmup)
base = cs.run(cs.LRU(a.cap), test, a.warmup)
print(f"test accesses {n}, lru miss {base / n:.2%}")
for h in (500, 1000, 2000, 3000, 4000):
    hot = {k for k, _ in freq.most_common(h)}
    m = cs.run(PinnedLRU(a.cap, hot), test, a.warmup)
    print(f"  pin {h:5}: miss {m / n:.2%}  vs lru {m / base - 1:+.1%}")
