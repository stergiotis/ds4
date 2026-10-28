# rocm/gen: kernels from the dskgen plans

The headers here are written by `dskgen emit-plan` in the
[ds4_parametric](https://github.com/stergiotis/ds4_parametric) repository
(`dskgen/`), one per kernel family, from the plans its chooser wrote over the
measured schedule spaces. Do not edit them: change the plan, the space or
the skeleton there and emit again.

Each header holds a family's chosen kernels (self-contained, in
`namespace dskgen`, each pinning the floating-point semantics of the fork
kernel it matches bit for bit with `#pragma clang fp`, since this tree sets
them per file), a launcher `dskgen::<family>_launch(plan, K, ...)`
keyed by plan and token count, and at the end the register contract: the
VGPRs and occupancy each kernel was measured at. The engine calls the
launcher before its own kernel in `moe_mv_dispatch` (rocm/ds4_rocm_qwen4.cuh)
and `ds4_gpu_kolibri_moe_q4k_tensor` (rocm/ds4_rocm_kolibri.cuh); a
launcher that has no kernel for the call returns false and the hand-written
kernel runs. `DS4_DSKGEN=0` leaves the plan kernels out.

| Header | Family | Plans | Launched for |
|---|---|---|---|
| `dskgen_moe_q4k_rows.cuh` | Q4_K expert gate/up, Q8_0 shared expert | exact | Qwen3.8 decode, T = 1..3 |
| `dskgen_moe_down_mxfp4_rows.cuh` | MXFP4 expert down, Q8_0 shared expert | exact; budget 5e-3 (block-int8 dots, the `DS4_QWEN4_MXFP4_Q8` default) | Qwen3.8 decode, T = 1..3 |

`make rocm-contract` compiles `ds4_rocm.cu` once more with
`-Rpass-analysis=kernel-resource-usage` and runs `dskgen-contract.sh`
(also written by `emit-plan`) over the remarks: every plan kernel must come
out of this compiler with the registers and occupancy it was measured at,
or the check fails and names it. A kernel the production compiler
allocates differently from the measured one is not what the plan chose.

To regenerate, in ds4_parametric/dskgen:

```sh
cargo run --release -- emit-plan ../../ds4/rocm/gen specs/plan-moe-q4k-rows-strix-halo-gfx1151.toml
cargo run --release -- emit-plan ../../ds4/rocm/gen \
    specs/plan-moe-down-mxfp4-rows-strix-halo-gfx1151.toml \
    specs/plan-moe-down-mxfp4-rows-strix-halo-gfx1151-budget.toml
```
