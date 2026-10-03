#!/usr/bin/env python3
"""Golden fixtures from the numpy reference (kolibri_ref.py).

    uv run make_golden.py --model-dir SNAPSHOT [--out ../golden/kolibri] [--gen 16]

For every prompt this writes <name>.npz with:
  tokens          [T]            prompt ids (HF tokenizers)
  resid_first     [L+1, H] f32   residual stream at position 0 after embedding / each layer
  resid_last      [L+1, H] f32   same at the last prompt position
  resid_rms       [L+1, T] f32   RMS of the residual at every position
  experts         [L, T, 6] i16  selected routed experts (selection order)
  expert_w        [L, T, 6] f32  their sigmoid weights
  top_ids/top_val [T, 32]        prefill top-32 logits at every position
  last_logits     [V] f32        full logits at the last prompt position
  gen             [G]            greedy continuation
  gen_top_ids/val [G, 8]         top-8 logits at each decode step
and index.json summarising prompts, texts and greedy continuations.
"""

import argparse
import json
import os
import time

import numpy as np
from tokenizers import Tokenizer

from kolibri_ref import Kolibri

HERE = os.path.dirname(os.path.abspath(__file__))


def prompts(tok):
    golden = json.load(open(os.path.join(HERE, "..", "golden", "kolibri", "tokenizer.json")))
    chat = {c["name"]: c for c in golden["chat"]}
    raw = [
        ("en_capital", "The capital of France is"),
        ("de_story", "Es war einmal ein kleiner Kolibri, der"),
        ("count", "1, 2, 3, 4, 5,"),
    ]
    out = [(n, t, tok.encode(t, add_special_tokens=False).ids) for n, t in raw]
    for n in ("chat_en_default", "chat_de_nothink"):
        out.append((n, chat[n]["rendered"], chat[n]["ids"]))
    return out


def topk(x, k):
    ids = np.argsort(-x, axis=-1, kind="stable")[..., :k]
    return ids.astype(np.int32), np.take_along_axis(x, ids, -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "golden", "kolibri"))
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--act-quant", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    model = Kolibri(args.model_dir, act_quant=args.act_quant)
    index_path = os.path.join(args.out, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    for name, text, ids in prompts(tok):
        if args.only and name not in args.only.split(","):
            continue
        t0 = time.time()
        trace = {}
        gen, lg, steps = model.greedy(ids, args.gen, trace)
        resid = np.stack(trace["resid"])                    # [L+1, T, H]
        ti, tv = topk(lg, 32)
        gi, gv = topk(np.stack(steps), 8)
        np.savez_compressed(
            os.path.join(args.out, name + ".npz"),
            tokens=np.array(ids, np.int32),
            resid_first=resid[:, 0], resid_last=resid[:, -1],
            resid_rms=np.sqrt((resid ** 2).mean(-1)),
            experts=np.stack(trace["experts"]).astype(np.int16),
            expert_w=np.stack(trace["weights"]).astype(np.float32),
            top_ids=ti, top_val=tv, last_logits=lg[-1].astype(np.float32),
            gen=np.array(gen, np.int32), gen_top_ids=gi, gen_top_val=gv)
        index[name] = {"prompt": text, "n_prompt": len(ids), "gen_ids": gen,
                       "gen_text": tok.decode(gen, skip_special_tokens=False),
                       "act_quant": args.act_quant}
        print(f"{name}: {len(ids)} tok, {time.time() - t0:.0f}s -> {index[name]['gen_text']!r}",
              flush=True)
        with open(index_path, "w") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
