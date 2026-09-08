# MLX K/V capacity experiments

Standalone Metal mechanism probes. These use the installed MLX C++ API, load no
model, and do not change the runner's default attention implementation.

`capacity_probe.cpp` checks BF16 `[1,2,T,256]` append against a compact concat
oracle. Its diagnostic reads scalar Metal buffer identity, offset and allocation
size without retaining a buffer owner. Non-growth appends must reuse storage;
held backing/view aliases must preserve their bits and force a separate buffer.
Growth, clamp, reset and optional scalar SDPA are covered. It does not prove
Swift ARC ownership or production allocation peaks.

`capacity_gqa_probe.cpp` compares padded capacity views with compact independent
inputs for Q24/KV2, GQA12 and D256. Fourteen lengths and 49 combinations cover
unmasked attention and boolean masks, including representative 512-block QSA
visibility plus the incomplete tail. The learned QSA indexer is not executed.
Last-row-only masks independently check the GQA head mapping. SDPA output is
checked bitwise; no absence of internal SDPA copies is claimed.

From the repository root, use the same pinned and locally patched MLX build as
the runner. Run only while no other owned model experiment uses the GPU:

```sh
CAPACITY_MLX_PREFIX="$(pwd)/../qwen38-ssd/runtime/mlx-serve/lib/mlx"
mkdir -p results/kv-capacity-local
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun clang++ -std=c++20 -O2 -Wall -Wextra -arch arm64 \
  -mmacosx-version-min=26.2 -fno-fast-math \
  -I"$CAPACITY_MLX_PREFIX/include" native/kv-capacity/capacity_probe.cpp \
  "$CAPACITY_MLX_PREFIX/lib/libmlx.dylib" \
  -Wl,-rpath,"$CAPACITY_MLX_PREFIX/lib" -Wl,-undefined,error \
  -framework Metal -framework Foundation \
  -o results/kv-capacity-local/capacity-probe
```

Build `capacity_gqa_probe.cpp` with the same flags and a separate output binary.
The append binary accepts `--async-eval` and `--sdpa`; the GQA binary requires
`--sdpa` and optionally `--async-eval`. Each emits NDJSON and exits nonzero on
failure. Use fresh logs and the experiment controller for bounded execution and
reference-service restoration. Do not weaken allocation assertions after a
failure: allocator reuse of oversized buffers can legitimately prevent donation.

The tested installed stamp is `mlx=1f8e74e3f12f mlxc=56b2d39fc831 target=26.2`,
header/runtime MLX version 0.32.2. Local patches matter; see the
[design and recorded results](../../docs/research/KV_ATTENTION_STORAGE_DESIGN.md).
The probes independently call public APIs; they do not copy an upstream kernel
or allocator implementation. MLX is MIT licensed.
