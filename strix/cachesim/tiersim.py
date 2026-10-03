#!/usr/bin/env python3
"""Two-tier (Q2/Q4) expert cache replay. Capacity is in Q2 units (a Q4
expert takes 2). A miss loads the Q2 copy (1 unit read); an entry is
upgraded to Q4 (2 units read) on its p-th hit while at Q2. Eviction is LRU
by entry. Reports read volume per token (in Q4-expert equivalents, 13.5 MiB)
and the share of expert uses computed at Q2. p=0 means Q4 always (baseline)."""
import collections, importlib.util, os, sys
spec = importlib.util.spec_from_file_location("cs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cachesim.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)

def run(steps, cap_q4, p, warmup):
    cap = 2 * cap_q4
    od = collections.OrderedDict()  # key -> [tier(1|2), hits_at_q2]
    used = 0
    read = uses = q2uses = 0
    for tok, layer, ex in steps:
        pinned = {(layer, e) for e in ex}
        for e in ex:
            k = (layer, e)
            meas = tok >= warmup
            r = 0
            if k in od:
                ent = od[k]; od.move_to_end(k)
                if ent[0] == 1:
                    ent[1] += 1
                    if p and ent[1] >= p:
                        ent[0] = 2; used += 1; r = 2
            else:
                ent = [2 if p == 0 else 1, 0]
                r = ent[0]
                od[k] = ent; used += ent[0]
            while used > cap:
                for v in od:
                    if v not in pinned:
                        used -= od.pop(v)[0]
                        break
            if meas:
                read += r; uses += 1; q2uses += od[k][0] == 1 if k in od else 0
    return read, uses, q2uses

steps = cs.load(sys.argv[1:])
warm = 300
ntok = steps[-1][0] + 1 - warm
for cap in (4790, 5869):
    print(f"cap = {cap} Q4 slots ({cap * 13.5 / 1024:.1f} GiB)")
    base = None
    for p in (0, 1, 2, 3, 5, 10, 10**9):
        read, uses, q2 = run(steps, cap, p, warm)
        per_tok = read / 2 / ntok  # in Q4-expert units
        if base is None: base = per_tok
        label = "Q4 only" if p == 0 else ("Q2 only" if p == 10**9 else f"upgrade at hit {p}")
        print(f"  {label:17} reads/token {per_tok:6.1f} Q4-eq ({per_tok * 13.5 / 1024:5.2f} GiB)  vs Q4-only {per_tok / base - 1:+6.1%}  uses at Q2 {q2 / uses:6.1%}")
