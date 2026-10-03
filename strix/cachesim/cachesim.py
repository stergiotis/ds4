#!/usr/bin/env python3
"""Replay GLM routing traces against expert-cache policies.

A trace line is "pos layer e0 .. e7 [; w0 .. w7]" (DS4_GLM_ROUTE_TRACE). Every expert is
the same size, so capacity is counted in experts. The experts of the layer
being routed are pinned: a policy may not evict one of them to make room for
another. Misses are counted after a warm-up of the first --warmup tokens.

  cachesim.py [--cap N ...] [--warmup T] TRACE [TRACE ...]

Several traces are replayed back to back, as one server would serve them.
"""
import argparse
import collections
import sys


def load(paths):
    steps = []  # (token_seq, layer, tuple(experts))
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
                ids = v[2:v.index(";")] if ";" in v else v[2:]
                steps.append((tok, layer, tuple(int(x) for x in ids)))
        tok += 1
    return steps


class Policy:
    def __init__(self, cap):
        self.cap = cap

    def step(self, layer, keys):
        """Access all keys of one layer step; return the number of misses."""
        pinned = set(keys)
        misses = 0
        for k in keys:
            if self.hit(k):
                continue
            misses += 1
            self.insert(k, layer, pinned)
        return misses


class LRU(Policy):
    """Global LRU; with past_first, prefer victims from layers <= the current
    one (ds4's DS4_ROCM_STREAM_EVICT_PAST_LAYERS_FIRST)."""

    def __init__(self, cap, past_first=False):
        super().__init__(cap)
        self.past_first = past_first
        self.by_layer = collections.defaultdict(collections.OrderedDict)
        self.clock = 0
        self.n = 0

    def hit(self, k):
        od = self.by_layer[k[0]]
        if k in od:
            self.clock += 1
            od[k] = self.clock
            od.move_to_end(k)
            return True
        return False

    def victim(self, layer, pinned, past_only):
        best = None
        for l, od in self.by_layer.items():
            if past_only and l > layer:
                continue
            for k, t in od.items():
                if k in pinned:
                    continue
                if best is None or t < best[1]:
                    best = (k, t)
                break
        return best

    def insert(self, k, layer, pinned):
        if self.n >= self.cap:
            v = self.victim(layer, pinned, True) if self.past_first else None
            if v is None:
                v = self.victim(layer, pinned, False)
            del self.by_layer[v[0][0]][v[0]]
            self.n -= 1
        self.clock += 1
        self.by_layer[k[0]][k] = self.clock
        self.n += 1


class S3FIFO(Policy):
    """S3-FIFO (Yang et al., SOSP'23): small FIFO S (10%), main FIFO M with
    lazy reinsertion, ghost FIFO G of keys evicted from S."""

    def __init__(self, cap, small=0.1, promote=1):
        super().__init__(cap)
        self.s_cap = max(1, int(cap * small))
        self.m_cap = cap - self.s_cap
        self.promote = promote
        self.S = collections.OrderedDict()  # key -> freq, head = newest (end)
        self.M = collections.OrderedDict()
        self.G = collections.OrderedDict()

    def hit(self, k):
        for q in (self.S, self.M):
            if k in q:
                q[k] = min(q[k] + 1, 3)
                return True
        return False

    def evict_m(self, pinned):
        while True:
            k, f = next(iter(self.M.items()))
            del self.M[k]
            if f > 0 or k in pinned:
                self.M[k] = max(f - 1, 0)
                continue
            return

    def evict_s(self, pinned):
        while self.S:
            k, f = next(iter(self.S.items()))
            del self.S[k]
            if f >= self.promote or k in pinned:
                if len(self.M) >= self.m_cap:
                    self.evict_m(pinned)
                self.M[k] = 0
                continue
            self.G[k] = None
            if len(self.G) > self.m_cap:
                self.G.popitem(last=False)
            return

    def insert(self, k, layer, pinned):
        if k in self.G:
            del self.G[k]
            if len(self.M) >= self.m_cap:
                self.evict_m(pinned)
            self.M[k] = 0
            return
        if len(self.S) >= self.s_cap:
            self.evict_s(pinned)
        self.S[k] = 0


class WTinyLFU(Policy):
    """W-TinyLFU (Einziger et al.): LRU window (1%), SLRU main (20% probation,
    80% protected), admission by frequency. Frequencies are exact counts
    halved every 10 x cap accesses (an ideal sketch)."""

    def __init__(self, cap, window=0.01):
        super().__init__(cap)
        self.w_cap = max(1, int(cap * window))
        main = cap - self.w_cap
        self.prot_cap = int(main * 0.8)
        self.prob_cap = main - self.prot_cap
        self.W = collections.OrderedDict()
        self.prob = collections.OrderedDict()
        self.prot = collections.OrderedDict()
        self.freq = collections.Counter()
        self.ops = 0
        self.reset_at = 10 * cap

    def count(self, k):
        self.freq[k] += 1
        self.ops += 1
        if self.ops >= self.reset_at:
            self.ops = 0
            for key in list(self.freq):
                self.freq[key] //= 2
                if not self.freq[key]:
                    del self.freq[key]

    def hit(self, k):
        self.count(k)
        if k in self.W:
            self.W.move_to_end(k)
            return True
        if k in self.prot:
            self.prot.move_to_end(k)
            return True
        if k in self.prob:
            del self.prob[k]
            self.prot[k] = None
            if len(self.prot) > self.prot_cap:
                d, _ = self.prot.popitem(last=False)
                self.prob[d] = None
            return True
        return False

    def first_unpinned(self, q, pinned):
        for k in q:
            if k not in pinned:
                return k
        return None

    def insert(self, k, layer, pinned):
        self.W[k] = None
        if len(self.W) <= self.w_cap:
            return
        cand = self.first_unpinned(self.W, pinned)
        if cand is None:
            return
        del self.W[cand]
        if len(self.prob) + len(self.prot) < self.prob_cap + self.prot_cap:
            self.prob[cand] = None
            return
        victim = self.first_unpinned(self.prob, pinned) or self.first_unpinned(self.prot, pinned)
        if victim is None or self.freq[cand] <= self.freq[victim]:
            return  # candidate rejected
        (self.prob if victim in self.prob else self.prot).pop(victim)
        self.prob[cand] = None


class ARC(Policy):
    """Adaptive Replacement Cache (Megiddo and Modha)."""

    def __init__(self, cap):
        super().__init__(cap)
        self.p = 0
        self.T1, self.T2 = collections.OrderedDict(), collections.OrderedDict()
        self.B1, self.B2 = collections.OrderedDict(), collections.OrderedDict()

    def hit(self, k):
        if k in self.T1:
            del self.T1[k]
            self.T2[k] = None
            return True
        if k in self.T2:
            self.T2.move_to_end(k)
            return True
        return False

    def replace(self, in_b2, pinned):
        use_t1 = self.T1 and (len(self.T1) > self.p or (in_b2 and len(self.T1) == self.p))
        for src, ghost in ((self.T1, self.B1), (self.T2, self.B2)) if use_t1 else \
                ((self.T2, self.B2), (self.T1, self.B1)):
            for k in src:
                if k not in pinned:
                    del src[k]
                    ghost[k] = None
                    return

    def insert(self, k, layer, pinned):
        c = self.cap
        if k in self.B1:
            self.p = min(c, self.p + max(len(self.B2) // max(len(self.B1), 1), 1))
            self.replace(False, pinned)
            del self.B1[k]
            self.T2[k] = None
            return
        if k in self.B2:
            self.p = max(0, self.p - max(len(self.B1) // max(len(self.B2), 1), 1))
            self.replace(True, pinned)
            del self.B2[k]
            self.T2[k] = None
            return
        l1 = len(self.T1) + len(self.B1)
        if l1 == c:
            if len(self.T1) < c:
                self.B1.popitem(last=False)
                self.replace(False, pinned)
            else:
                for q in self.T1:
                    if q not in pinned:
                        del self.T1[q]
                        break
        elif l1 < c and l1 + len(self.T2) + len(self.B2) >= c:
            if l1 + len(self.T2) + len(self.B2) == 2 * c:
                self.B2.popitem(last=False)
            self.replace(False, pinned)
        self.T1[k] = None


def belady(steps, cap, warmup):
    """Optimal eviction: evict the resident expert used furthest in future."""
    import heapq
    INF = float("inf")
    seq = [(i, (layer, e)) for i, (_, layer, ex) in enumerate(steps) for e in ex]
    nxt = [INF] * len(seq)
    last = {}
    for j in range(len(seq) - 1, -1, -1):
        s, k = seq[j]
        nxt[j] = last.get(k, INF)
        last[k] = s
    # next-use step index for each access position
    resident = {}  # key -> next use step
    heap = []  # (-next_use, key)
    misses = 0
    j = 0
    for i, (tok, layer, ex) in enumerate(steps):
        pinned = {(layer, e) for e in ex}
        for e in ex:
            k = (layer, e)
            nu = nxt[j]
            j += 1
            if k in resident:
                resident[k] = nu
                heapq.heappush(heap, (-nu, k))
                continue
            if tok >= warmup:
                misses += 1
            if len(resident) >= cap:
                held = []
                while heap:
                    negnu, v = heapq.heappop(heap)
                    if v not in resident or resident[v] != -negnu:
                        continue
                    if v in pinned:
                        held.append((negnu, v))
                        continue
                    del resident[v]
                    break
                for h in held:
                    heapq.heappush(heap, h)
            resident[k] = nu
            heapq.heappush(heap, (-nu, k))
    return misses


def run(policy, steps, warmup):
    misses = 0
    for tok, layer, ex in steps:
        m = policy.step(layer, [(layer, e) for e in ex])
        if tok >= warmup:
            misses += m
    return misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, nargs="+", default=[5301, 5877])
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--policies", default="lru,lru-past,s3fifo,s3fifo-p2,wtinylfu,arc,opt")
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args()
    steps = load(a.traces)
    ntok = steps[-1][0] + 1 if steps else 0
    keys = {(l, e) for _, l, ex in steps for e in ex}
    measured = [s for s in steps if s[0] >= a.warmup]
    n_acc = sum(len(s[2]) for s in measured)
    print(f"traces={len(a.traces)} tokens={ntok} layer_steps={len(steps)} "
          f"distinct_experts={len(keys)} measured_accesses={n_acc} warmup_tokens={a.warmup}")
    makers = {
        "lru": lambda c: LRU(c),
        "lru-past": lambda c: LRU(c, past_first=True),
        "s3fifo": lambda c: S3FIFO(c),
        "s3fifo-p2": lambda c: S3FIFO(c, promote=2),
        "wtinylfu": lambda c: WTinyLFU(c),
        "arc": lambda c: ARC(c),
    }
    for cap in a.cap:
        base = None
        print(f"\ncap={cap} experts")
        for name in a.policies.split(","):
            if name == "opt":
                m = belady(steps, cap, a.warmup)
            else:
                m = run(makers[name](cap), steps, a.warmup)
            if base is None:
                base = m
            print(f"  {name:10} miss rate {m / n_acc:7.2%}  misses/token {m / max(ntok - a.warmup, 1):7.2f}"
                  f"  vs {a.policies.split(',')[0]} {m / base - 1:+7.1%}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
