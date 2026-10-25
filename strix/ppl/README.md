# Perplexity text set for quant comparisons

Two fixed texts for comparing quantizations of one model (GLM-5.3-Flash Q2
against Q4, a Q2 tier against plain Q4) token by token. They replace the 1,500
scored Go and Markdown tokens that were lost with `~/.local/share/ds4/evals/`.
Score each file separately, so Go and Markdown agreement stay apart.

| File | Source | GLM-5.3 tokens |
|---|---|---:|
| `go.txt` | Go 1.27.0 `src/encoding/csv/reader.go`, verbatim (sha256 `c37960d7...9fef1e`) | 3,936 |
| `md.txt` | `strix/README.md` lines 1-125 at 4be83d5 (prose and tables, written after the model's training data) | 2,810 |

The stdlib file is code the model has probably seen; the Markdown file it
cannot have. Only the first 1,032 tokens are used: 32 as the fixed prefix, then
1,000 scored (`-n 1000`).

```sh
for f in go md; do
    DS4_PPL_DUMP=$OUT/$f.dump ./ds4 --rocm -m MODEL.gguf --ctx 2048 -n 1000 \
        --perplexity-file strix/ppl/$f.txt
done
strix/pplcompare.py $REF/go.dump $OUT/go.dump
```

Add the run's own flags (`--ssd-streaming`, DS4_* settings) after `--rocm`.
The dumps depend on the build and on memory pressure (strix/README.md), so
compare runs from the same build and settings.
