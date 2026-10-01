#!/usr/bin/env python3
"""Weight-aware Q2/Q4 tiering replay over traces with gate weights.

Capacity is counted in Q2 units (a Q4 expert takes 2), LRU by entry, the
current layer's experts pinned. Policies:

  reuse:P    a miss loads the Q2 copy; the P-th hit at Q2 upgrades it to Q4
  weight:W   a miss with gate weight < W loads Q2, otherwise Q4; a Q2 entry
             that is later selected with weight >= W is upgraded to Q4

Reports reads per token in Q4-expert units (13.5 MiB) against Q4-only LRU,
and the share of expert uses and of gate-weight mass computed at Q2.

  tiersim_weighted.py [--cap N] [--warmup T] TRACE [TRACE ...]
"""
import argparse
import collections


def load(paths):
    steps, tok = [], 0
    for path in paths:
        last = None
        with open(path) as f:
            for line in f:
                v = line.split()
                if ";" not in v:
                    raise SystemExit(f"{path}: no gate weights; trace with a newer ds4")
                i = v.index(";")
                pos, layer = int(v[0]), int(v[1])
                if last is not None and pos != last:
                    tok += 1
                last = pos
                steps.append((tok, layer, [int(x) for x in v[2:i]], [float(x) for x in v[i + 1:]]))
        tok += 1
    return steps


def run(steps, cap_q4, mode, arg, warmup):
    cap = 2 * cap_q4
    od = collections.OrderedDict()  # key -> [tier, hits_at_q2]
    used = read = uses = q2uses = 0
    wsum = q2w = 0.0
    for tok, layer, ex, ws in steps:
        pinned = {(layer, e) for e in ex}
        for e, w in zip(ex, ws):
            k = (layer, e)
            r = 0
            if k in od:
                ent = od[k]
                od.move_to_end(k)
                if ent[0] == 1:
                    ent[1] += 1
                    up = (mode == "reuse" and ent[1] >= arg) or (mode == "weight" and w >= arg)
                    if up:
                        ent[0] = 2
                        used += 1
                        r = 2
            else:
                if mode == "q4":
                    tier = 2
                elif mode == "reuse":
                    tier = 1
                else:
                    tier = 1 if w < arg else 2
                od[k] = ent = [tier, 0]
                used += tier
                r = tier
            while used > cap:
                for v in od:
                    if v not in pinned:
                        used -= od.pop(v)[0]
                        break
            if tok >= warmup:
                read += r
                uses += 1
                wsum += w
                if ent[0] == 1:
                    q2uses += 1
                    q2w += w
    return read, uses, q2uses, wsum, q2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, nargs="+", default=[5244, 5869])
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args()
    steps = load(a.traces)
    ntok = steps[-1][0] + 1 - a.warmup
    allw = sorted(w for _, _, _, ws in steps for w in ws)
    q = lambda p: allw[int(p * (len(allw) - 1))]
    print(f"gate weight quantiles: p10 {q(.1):.3f} p25 {q(.25):.3f} p50 {q(.5):.3f} p75 {q(.75):.3f} p90 {q(.9):.3f}")
    for cap in a.cap:
        print(f"\ncap = {cap} Q4 slots")
        base = None
        configs = [("q4", 0)] + [("reuse", p) for p in (2, 3)] + \
                  [("weight", q(p)) for p in (0.25, 0.5, 0.75)]
        for mode, arg in configs:
            read, uses, q2u, wsum, q2w = run(steps, cap, mode, arg, a.warmup)
            per_tok = read / 2 / ntok
            if base is None:
                base = per_tok
            label = {"q4": "Q4 only", "reuse": f"reuse, upgrade at hit {arg}",
                     "weight": f"weight < {arg:.3f}"}[mode]
            print(f"  {label:26} reads/token {per_tok:6.1f}  vs Q4-only {per_tok / base - 1:+6.1%}"
                  f"  uses at Q2 {q2u / uses:6.1%}  weight mass at Q2 {q2w / wsum:6.1%}")


if __name__ == "__main__":
    main()
