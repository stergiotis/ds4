#!/usr/bin/env python3
"""Compare two DS4_PPL_DUMP files token by token.

Each line: "target logprob id:logprob ... (top 20)". Reports mean NLL of each
run, the NLL change, top-1 agreement, and KL(ref || test) over the reference's top-20
tokens plus one bucket for the rest (see kl()).

Before that summary line it prints the same by quartile of reference entropy
(over the top-20 plus the rest bucket), easiest first, with the mean top-1
probability of each run, which shows whether the test run is over-confident
where the reference is unsure.

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
    """KL(ref || test) over one partition: the reference's top-20 tokens and
    one bucket for the rest. A token missing from the test top-20 gets an even
    share of the test rest, capped at its smallest listed probability. Every
    bucket has a mass in both runs, so the result is a lower bound of the
    full KL and never negative."""
    qrest = max(1e-12, 1.0 - sum(math.exp(v) for v in q.values()))
    missing = [k for k in p if k not in q]
    fill = min(math.exp(min(q.values())), qrest / max(1, len(missing))) if missing else 0.0
    qk = {k: math.exp(q[k]) if k in q else fill for k in p}
    pk = {k: math.exp(v) for k, v in p.items()}
    prest = max(1e-12, 1.0 - sum(pk.values()))
    qo = max(1e-12, 1.0 - sum(qk.values()))
    return sum(pk[k] * math.log(pk[k] / max(1e-30, qk[k])) for k in p) + prest * math.log(prest / qo)


def entropy(p):
    rest = max(1e-12, 1.0 - sum(math.exp(v) for v in p.values()))
    return -sum(math.exp(v) * v for v in p.values()) - rest * math.log(rest)


def top1p(p):
    return math.exp(max(p.values()))


ref, test = read(sys.argv[1]), read(sys.argv[2])
n = min(len(ref), len(test))
assert all(ref[i][0] == test[i][0] for i in range(n)), "different token streams"
nll_r = -sum(r[1] for r in ref[:n]) / n
nll_t = -sum(t[1] for t in test[:n]) / n
top1 = sum(max(r[2], key=r[2].get) == max(t[2], key=t[2].get) for r, t in zip(ref[:n], test[:n])) / n
pairs = sorted(zip(ref[:n], test[:n]), key=lambda rt: entropy(rt[0][2]))
print("quartile  entropy      KL mean  nll ref  nll delta  top1 agree  top1 p ref  test")
for q in range(4):
    part = pairs[q * n // 4:(q + 1) * n // 4]
    m = len(part)
    print(f"Q{q + 1}        {entropy(part[0][0][2]):.2f}-{entropy(part[-1][0][2]):<5.2f}  "
          f"{sum(kl(r[2], t[2]) for r, t in part) / m:.5f}  "
          f"{-sum(r[1] for r, _ in part) / m:7.4f}  {sum(r[1] - t[1] for r, t in part) / m:+9.4f}  "
          f"{sum(max(r[2], key=r[2].get) == max(t[2], key=t[2].get) for r, t in part) / m:10.2%}  "
          f"{sum(top1p(r[2]) for r, _ in part) / m:10.4f}  {sum(top1p(t[2]) for _, t in part) / m:.4f}")
kls = sorted(kl(r[2], t[2]) for r, t in zip(ref[:n], test[:n]))
mean_kl = sum(kls) / n
print(f"tokens {n}  nll ref {nll_r:.4f} test {nll_t:.4f} (delta {nll_t - nll_r:+.4f}, ppl x{math.exp(nll_t - nll_r):.4f})"
      f"  top1 agree {top1:.2%}  KL mean {mean_kl:.5f} p50 {kls[n // 2]:.5f} p99 {kls[int(n * .99)]:.4f} max {kls[-1]:.3f}")
