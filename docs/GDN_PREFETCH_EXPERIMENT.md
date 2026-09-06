# GDN QKV load lookahead experiment

The existing GDN GEMV probe shows a large difference between the QKV projection
and the output head's logical weight bytes per elapsed GPU time. Earlier BM
geometry scans did not establish a durable full-model gain. This experiment
changes how far ahead a thread requests BF16 data, keeping the original
arithmetic and thread mapping.

`native/gdn_prefetch.metal` preloads four consecutive K tiles before consuming
them. Mode `prefetch4` uses scalar BF16 loads; mode `prefetch4Vector` requests
aligned groups of four BF16 values. Both keep MLX's
BM8/BN1/SM1/SN32/TM4/TN4 geometry, tile/row/scalar accumulation order, and
16/8/4/2/1 shuffle reduction. The compiler still determines actual load
scheduling; these are source-level candidates, not evidence of asynchronous
hardware prefetch or higher bandwidth.

The dispatcher only selects the new kernels for a single-vector, original
BF16 QKV multiplication with K=2560 and N=10240, dense matrix strides, aligned
offsets, and no axpby. Other shapes, prefill GEMM, batched inputs, and modes 0–6
retain their existing implementation. Selection defaults to mode 0.

## Build and quick gate

The helper consumes the existing isolated GEMM and BM1/2 build stages. It
compiles one new Metal object and a copied dispatcher, then links a fresh MLX
stage. Original runtime files and link inputs are hash checked and remain
unchanged. It does not load a model, start timing, or run a GPU workload.

```bash
python3 scripts/build_mlx_gdn_prefetch.py \
  --output-root results/manual-gdn-prefetch-native
```

After building the updated Swift CLI and obtaining exclusive ownership of the
GPU/model workload, the existing four-layer, real-weight probe can compare both
candidates with mode 0 in one process:

```bash
env DYLD_LIBRARY_PATH="$PWD/results/manual-gdn-prefetch-native/lib" \
  .build/release/ane-runner probe-gpu-matvec \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --gdn-gemv-order reference,prefetch4,prefetch4Vector \
  --repeats 64 \
  --output results/manual-gdn-prefetch-matvec.json
```

The probe rotates layers 0/4/8/12 and changes candidate order between iterations.
It requires finite, bitwise-equal BF16 outputs by default. It also checks a
native counter outside each timed window: QKV must dispatch the selected new
kernel exactly once; Z, out, and head must not dispatch it. The selector resets
to mode 0 on exit. Native GPU timing can be requested separately using the
existing `--gpu-command-timing-output` flag; sampled results must be identified
separately from normal performance measurements.

Compilation alone does not validate outputs or performance. Full-model decode
testing is worthwhile only after this quick gate shows a repeatable benefit.
The variants remain optional until that evidence exists. More live registers
may reduce occupancy, so load lookahead can also be slower than the baseline.

## First local result, 2026-09-07

The final isolated stage `results/gdn-prefetch-v1/native-v2` compiled and linked;
the original inputs remained unchanged. Swift release build passed. The
four-layer probe completed with its finite/bitwise gate and dispatch assertions.
Inputs here are deterministic synthetic nonzero BF16 vectors against real
weights, not captured model activations or an end-to-end model acceptance.

| QKV mode | Median new-graph + evaluation wall time | Throughput change against reference |
| --- | ---: | ---: |
| reference | 0.2480 ms | — |
| prefetch4 | 0.2510 ms | -1.17% |
| prefetch4Vector | 0.2531 ms | -1.98% |

Each mode has 64 measured calls, rotating four real QKV matrices. Non-QKV
cases dispatch the original kernel and therefore serve as unchanged-path
controls; their timing differences are not gains from this candidate.
These are host wall intervals including graph construction and evaluation wait,
not device-only intervals or measured DRAM bandwidth.

Neither candidate met the local performance screen, so this version was not
promoted to a full-model decode benchmark or to the default runtime. The
reference service was restored and verified with MTP/drafter disabled. Raw
records, compact medians and process lifecycle are in
`results/gdn-prefetch-v1/matvec.json`, `summary.json`, and `run-ledger.json`.

## Related source

The thread mapping and arithmetic are derived from the pinned MLX `gemv.h`;
the repository's upstream attribution applies. Independent investigation of
[DwarfStar's BF16 decode kernel](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/metal/glm53_bf16.metal#L29-L91)
found a similar technique: eight independent weight/input loads followed by
serial FMA consumption. That kernel uses a different lane mapping and
`simd_sum`; those parts are not substituted for MLX's reduction here. No
DwarfStar kernel source is copied into this experiment.
