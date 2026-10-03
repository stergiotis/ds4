#!/usr/bin/env python3
"""Compare two DS4_PPL_DUMP files token by token.

Each line: "target logprob id:logprob ... (top 20)". Reports mean NLL of each
run, the NLL change, top-1 agreement, and KL(ref || test) over the union of
both top-20 lists, with the rest of each distribution lumped into one bucket.

  pplcompare.py REF.dump TEST.dump
"""
import math
import sys


def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            v = line.split()
            top = {}
            for kv in v[2:]:
                k, p = kv.split(":")
                top[int(k)] = float(p)
            rows.append((int(v[0]), float(v[1]), top))
    return rows


def kl(p, q):
    keys = set(p) | set(q)
    floor = -30.0
    pp = {k: math.exp(p.get(k, floor)) for k in keys}
    qq = {k: math.exp(q.get(k, floor)) for k in keys}
    prest = max(1e-12, 1.0 - sum(math.exp(v) for v in p.values()))
    qrest = max(1e-12, 1.0 - sum(math.exp(v) for v in q.values()))
    out = prest * math.log(prest / qrest)
    for k in keys:
        if k in p:
            out += pp[k] * (p[k] - (q[k] if k in q else math.log(qrest)))
    return out


ref, test = read(sys.argv[1]), read(sys.argv[2])
n = min(len(ref), len(test))
assert all(ref[i][0] == test[i][0] for i in range(n)), "different token streams"
nll_r = -sum(r[1] for r in ref[:n]) / n
nll_t = -sum(t[1] for t in test[:n]) / n
top1 = sum(max(r[2], key=r[2].get) == max(t[2], key=t[2].get) for r, t in zip(ref[:n], test[:n])) / n
kls = sorted(kl(r[2], t[2]) for r, t in zip(ref[:n], test[:n]))
mean_kl = sum(kls) / n
print(f"tokens {n}  nll ref {nll_r:.4f} test {nll_t:.4f} (delta {nll_t - nll_r:+.4f}, ppl x{math.exp(nll_t - nll_r):.4f})"
      f"  top1 agree {top1:.2%}  KL mean {mean_kl:.5f} p50 {kls[n // 2]:.5f} p99 {kls[int(n * .99)]:.4f} max {kls[-1]:.3f}")
