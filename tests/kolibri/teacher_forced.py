#!/usr/bin/env python3
"""Teacher-forced agreement on a longer text, the measure that stays
meaningful when routing near-ties make greedy runs diverge.

    uv run teacher_forced.py ref    --model-dir SNAPSHOT --text FILE --out REF.npz [--act-quant]
    uv run teacher_forced.py ds4    --model GGUF --text FILE --out DS4.npz [--backend rocm|cpu]
    uv run teacher_forced.py compare REF.npz OTHER.npz [...]

Logits for every position are stored as float16.  compare reports, against
the first file: top-1 agreement, mean KL(ref || other) in nats, and each
file's perplexity on the text itself.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def tokens_for(model_dir_or_tok, text_path):
    tok = Tokenizer.from_file(os.path.join(model_dir_or_tok, "tokenizer.json"))
    return tok.encode(open(text_path).read(), add_special_tokens=False).ids


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    mode = sys.argv[1]
    args = dict(zip(sys.argv[2::2], sys.argv[3::2])) if mode != "compare" else {}
    if mode == "ref":
        from kolibri_ref import Kolibri
        ids = tokens_for(args["--model-dir"], args["--text"])
        m = Kolibri(args["--model-dir"], act_quant="--act-quant" in sys.argv)
        lg = m.logits(m.forward(ids))
        np.savez_compressed(args["--out"], tokens=np.array(ids, np.int32), logits=lg.astype(np.float16))
    elif mode == "ds4":
        snap = args.get("--tokenizer-dir", os.path.join(HERE, "..", "..", "..", "kolibri-ref"))
        ids = tokens_for(snap, args["--text"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "logits.f32")
            env = dict(os.environ, DS4_KOLIBRI_FT_TOKENS=",".join(map(str, ids)), DS4_KOLIBRI_FT_OUT=out)
            backend = args.get("--backend", "rocm")
            if backend == "rocm":
                env["DS4_KOLIBRI_GPU"] = "1"
            cmd = [os.path.join(HERE, "..", "..", "ds4"), "-m", args["--model"], "--first-token-test",
                   "--raw", "-p", "x", "--ctx", "8192", "--" + backend]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stderr[-2000:])
            lg = np.fromfile(out, np.float32).reshape(len(ids), -1)
        np.savez_compressed(args["--out"], tokens=np.array(ids, np.int32), logits=lg.astype(np.float16))
    else:
        files = sys.argv[2:]
        ref = np.load(files[0])
        ids = ref["tokens"]
        lr = log_softmax(ref["logits"])
        nxt = ids[1:]
        ppl = lambda lp: float(np.exp(-lp[np.arange(len(nxt)), nxt].mean()))
        print(f"{os.path.basename(files[0])}: {len(ids)} tokens, perplexity {ppl(lr[:-1]):.3f} (reference)")
        for f in files[1:]:
            o = np.load(f)
            assert (o["tokens"] == ids).all(), f
            lo = log_softmax(o["logits"])
            top1 = float((lo.argmax(-1) == lr.argmax(-1)).mean())
            kl = float((np.exp(lr) * (lr - lo)).sum(-1).mean())
            print(f"{os.path.basename(f)}: top1 agreement {top1:.4f}, mean KL {kl:.5f} nats, "
                  f"perplexity {ppl(lo[:-1]):.3f}")


if __name__ == "__main__":
    main()
