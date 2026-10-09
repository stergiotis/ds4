#!/usr/bin/env python3
"""Learned expert-cache eviction, evaluated offline on GLM routing traces.

The policy is LRB-style (Song et al., NSDI'20): on a miss with a full cache
it samples N resident experts, predicts each one's log distance to its next
use from features of the routing history, and evicts the furthest. It is
trained leave-one-domain-out: each trace in turn is the test domain and the
model learns only from the others.

Baselines on the same test trace and capacity: exact LRU, S3-FIFO and Belady
(from cachesim.py), sampled LRU (oldest of the same N samples, which isolates
the sampling) and the sampled oracle (true furthest next use among the N,
the ceiling for any predictor that sees the same samples).

  evictml.py [--cap 5869] [--warmup 300] [--samples 64] DOMAIN=TRACE[,TRACE] ...

Features per candidate: age, the last three reuse gaps, decayed access counts
(half-lives of 32, 256 and 2048 tokens), the count and rate since the trace
started (the request's own routing history), the candidate's layer and how
many layer steps until that layer is routed again, and the mean gate weight
of its recent uses when the trace records weights.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cachesim  # noqa: E402

HALF = (32.0, 256.0, 2048.0)
BIG = 1e7
FEATURES = ["log_age", "log_gap1", "log_gap2", "log_gap3", "ew32", "ew256", "ew2048",
            "count", "rate", "phase", "layer", "weight"]


def load_weighted(paths):
    """Like cachesim.load, plus the gate weights when a line records them."""
    steps, weights = [], []
    tok = 0
    for path in paths:
        last = None
        with open(path) as f:
            for line in f:
                v = line.split()
                if len(v) < 3:
                    continue
                pos, layer = int(v[0]), int(v[1])
                if last is not None and pos != last:
                    tok += 1
                last = pos
                if ";" in v:
                    i = v.index(";")
                    ids, w = v[2:i], [float(x) for x in v[i + 1:]]
                else:
                    ids, w = v[2:], None
                steps.append((tok, layer, tuple(int(x) for x in ids)))
                weights.append(w)
        tok += 1
    return steps, weights


class Trace:
    def __init__(self, paths, n_expert, layers):
        self.steps, self.weights = load_weighted(paths)
        self.n_expert = n_expert
        self.lidx = {l: i for i, l in enumerate(layers)}
        self.nl = len(layers)
        self.has_w = any(w is not None for w in self.weights)
        # next use (step index) of each access, in access order
        acc = [(s, self.key(layer, e)) for s, (_, layer, ex) in enumerate(self.steps) for e in ex]
        nxt = np.full(len(acc), BIG)
        last = {}
        for j in range(len(acc) - 1, -1, -1):
            s, k = acc[j]
            nxt[j] = last.get(k, BIG)
            last[k] = s
        self.next_use = nxt

    def key(self, layer, e):
        return self.lidx[layer] * self.n_expert + e


class State:
    """Per-key history features, updated on every access."""

    def __init__(self, n_keys, nl, n_expert):
        self.last = np.full(n_keys, -1.0)
        self.g = np.full((3, n_keys), BIG)
        self.cnt = np.zeros(n_keys)
        self.first_tok = np.full(n_keys, -1.0)
        self.ew = np.zeros((3, n_keys))
        self.ew_t = np.zeros((3, n_keys))
        self.wsum = np.zeros(n_keys)
        self.layer = (np.arange(n_keys) // n_expert).astype(float)
        self.nl = nl
        self.next_use = np.full(n_keys, BIG)

    def access(self, k, s, t, w, nu):
        last = self.last[k]
        g = self.g
        g[2, k] = g[1, k]
        g[1, k] = g[0, k]
        g[0, k] = s - last if last >= 0 else BIG
        self.last[k] = s
        if self.cnt[k] == 0:
            self.first_tok[k] = t
        self.cnt[k] += 1
        for i, h in enumerate(HALF):
            self.ew[i, k] = self.ew[i, k] * math.exp(-(t - self.ew_t[i, k]) / h) + 1.0
            self.ew_t[i, k] = t
        if w is not None:
            self.wsum[k] = 0.7 * self.wsum[k] + 0.3 * w
        self.next_use[k] = nu

    def features(self, c, s, t, cur_layer):
        f = np.empty((len(c), len(FEATURES)))
        f[:, 0] = np.log1p(s - self.last[c])
        f[:, 1:4] = np.log1p(self.g[:, c].T)
        for i, h in enumerate(HALF):
            f[:, 4 + i] = self.ew[i, c] * np.exp(-(t - self.ew_t[i, c]) / h)
        f[:, 7] = self.cnt[c]
        f[:, 8] = self.cnt[c] / (t - self.first_tok[c] + 1.0)
        f[:, 9] = (self.layer[c] - cur_layer) % self.nl
        f[:, 10] = self.layer[c]
        f[:, 11] = self.wsum[c]
        return f


def simulate(tr, cap, warmup, samples, choose, collect=None, seed=0):
    """Replay tr with a sampled policy. choose(feats, cands, state, s) returns
    the index of the victim. collect, if a list, receives (features, label)
    rows for a few candidates of every eviction. Returns misses after warmup."""
    rng = np.random.default_rng(seed)
    n_keys = tr.nl * tr.n_expert
    st = State(n_keys, tr.nl, tr.n_expert)
    slots = []
    where = {}
    misses = 0
    j = 0
    for s, (t, layer, ex) in enumerate(tr.steps):
        w = tr.weights[s]
        keys = [tr.key(layer, e) for e in ex]
        pinned = set(keys)
        cur = tr.lidx[layer]
        for i, k in enumerate(keys):
            nu = tr.next_use[j]
            j += 1
            if k not in where:
                if t >= warmup:
                    misses += 1
                if len(slots) >= cap:
                    c = rng.integers(0, len(slots), samples * 2)
                    cands = np.array([slots[x] for x in c if slots[x] not in pinned][:samples])
                    feats = st.features(cands, s, t, cur)
                    if collect is not None and t >= warmup:
                        pick = rng.choice(len(cands), min(16, len(cands)), replace=False)
                        dist = st.next_use[cands[pick]] - s
                        collect.append((feats[pick], np.log1p(np.minimum(dist, BIG))))
                    victim = int(cands[choose(feats, cands, st, s)])
                    x = where.pop(victim)
                    tail = slots.pop()
                    if x < len(slots):
                        slots[x] = tail
                        where[tail] = x
                where[k] = len(slots)
                slots.append(k)
            st.access(k, s, t, w[i] if w else None, nu)
    return misses


def choose_lru(feats, cands, st, s):
    return int(np.argmax(feats[:, 0]))


def choose_oracle(feats, cands, st, s):
    return int(np.argmax(st.next_use[cands]))


def choose_decayed(i):
    """Evict the lowest access count decayed with half-life HALF[i] (LRFU-like)."""
    def choose(feats, cands, st, s):
        return int(np.argmin(feats[:, 4 + i]))
    return choose


def measured_accesses(tr, warmup):
    return sum(len(ex) for t, _, ex in tr.steps if t >= warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=5869)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--no-weight", action="store_true", help="drop the gate-weight feature")
    ap.add_argument("--skip-exact", action="store_true", help="skip exact LRU/S3-FIFO/Belady")
    ap.add_argument("--objective", choices=["regression", "rank"], default="regression",
                    help="rank: LambdaRank over each eviction's sampled candidates")
    ap.add_argument("--dagger", type=int, default=0,
                    help="rounds of re-collecting training rows under the learned policy")
    ap.add_argument("--heuristics", action="store_true",
                    help="only replay the decayed-count heuristics, no model")
    ap.add_argument("domains", nargs="+", help="NAME=TRACE[,TRACE]")
    a = ap.parse_args()
    import lightgbm as lgb

    doms = [(d.split("=", 1)[0], d.split("=", 1)[1].split(",")) for d in a.domains]
    allsteps = [s for _, ps in doms for s in cachesim.load(ps)]
    layers = sorted({l for _, l, _ in allsteps})
    n_expert = max(e for _, _, ex in allsteps for e in ex) + 1
    traces = {name: Trace(ps, n_expert, layers) for name, ps in doms}
    print(f"layers={len(layers)} experts/layer={n_expert} cap={a.cap} samples={a.samples} "
          f"warmup={a.warmup}", flush=True)

    if a.heuristics:
        for name, tr in traces.items():
            n_acc = measured_accesses(tr, a.warmup)
            for i, h in enumerate(HALF):
                m = simulate(tr, a.cap, a.warmup, a.samples, choose_decayed(i))
                print(f"{name:10} decayed-{int(h):<6} {m / n_acc:9.2%}", flush=True)
        return
    rows = {}
    for name, tr in traces.items():
        col = []
        t0 = time.time()
        m = simulate(tr, a.cap, a.warmup, a.samples, choose_lru, collect=col)
        rows[name] = [col]
        print(f"collected {name}: {sum(len(c[1]) for c in col)} rows ({time.time() - t0:.0f}s)", flush=True)

    feat_mask = [i for i, f in enumerate(FEATURES) if not (a.no_weight and f == "weight")]
    print(f"\n{'test':10} {'policy':14} {'miss rate':>9} {'vs S3-FIFO':>10}")
    for name, tr in traces.items():
        n_acc = measured_accesses(tr, a.warmup)
        res = {}
        if not a.skip_exact:
            steps = tr.steps
            res["lru"] = cachesim.run(cachesim.LRU(a.cap), steps, a.warmup)
            res["s3fifo"] = cachesim.run(cachesim.S3FIFO(a.cap), steps, a.warmup)
            res["belady"] = cachesim.belady(steps, a.cap, a.warmup)
        res["sampled-lru"] = simulate(tr, a.cap, a.warmup, a.samples, choose_lru)
        res["sampled-oracle"] = simulate(tr, a.cap, a.warmup, a.samples, choose_oracle)
        train_names = [o for o in traces if o != name]

        def fit(groups):
            X = np.concatenate([g[0] for g in groups])[:, feat_mask]
            if a.objective == "rank":
                # relevance: rank of the next-use distance within the eviction
                y = np.concatenate([np.argsort(np.argsort(g[1])) for g in groups])
                ds = lgb.Dataset(X, y, group=[len(g[1]) for g in groups])
                params = {"objective": "lambdarank", "lambdarank_truncation_level": 4,
                          "label_gain": list(range(64))}
            else:
                ds = lgb.Dataset(X, np.concatenate([g[1] for g in groups]))
                params = {"objective": "regression"}
            params.update({"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
                           "feature_fraction": 0.9, "verbose": -1, "num_threads": 8})
            return lgb.train(params, ds, num_boost_round=a.trees)

        groups = [g for o in train_names for col in rows[o] for g in col]
        booster = fit(groups)
        pred_params = {"num_threads": 1}

        def choose_model(feats, cands, st, s):
            return int(np.argmax(booster.predict(feats[:, feat_mask], **pred_params)))

        for r in range(a.dagger):
            for o in train_names:
                col = []
                simulate(traces[o], a.cap, a.warmup, a.samples, choose_model, collect=col, seed=r + 1)
                groups.extend(col)
            booster = fit(groups)
            print(f"{name:10} dagger round {r + 1}: {len(groups)} evictions", flush=True)

        t0 = time.time()
        res["learned"] = simulate(tr, a.cap, a.warmup, a.samples, choose_model)
        dt = time.time() - t0
        base = res.get("s3fifo")
        for p, m in res.items():
            rel = f"{m / base - 1:+.1%}" if base else ""
            print(f"{name:10} {p:14} {m / n_acc:9.2%} {rel:>10}", flush=True)
        imp = booster.feature_importance("gain")
        top = sorted(zip(imp, [FEATURES[i] for i in feat_mask]), reverse=True)[:5]
        print(f"{name:10} learned replay {dt:.0f}s; top features: "
              + ", ".join(f"{n} {v / imp.sum():.0%}" for v, n in top), flush=True)


if __name__ == "__main__":
    main()
