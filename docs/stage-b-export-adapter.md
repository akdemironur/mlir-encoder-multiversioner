# Stage B Export Adapter

Do not teach `shortseq-specialize` to import models. A fixed adapter should map
one EmbeddingGemma-like export into the Stage B ABI documented in
`docs/stage-b-contract.md`.

## Boundary

Outside the pass: token IDs, tokenization, embedding lookup, masks, padding,
model loading, and source parameter names.

Inside the pass: pre-embedded `tensor<1x?x64xf32>` activations, ten immutable
parameter operands, exact-length dispatch, and generic fallback.

The adapter owns this operand order:

```text
q_w        tensor<64x64xf32>
k_w        tensor<64x64xf32>
v_w        tensor<64x64xf32>
o_w        tensor<64x64xf32>
norm_scale tensor<64xf32>
norm_bias  tensor<64xf32>
ff_w1      tensor<64x256xf32>
ff_b1      tensor<256xf32>
ff_w2      tensor<256x64xf32>
ff_b2      tensor<64xf32>
```

## Current Fake Parts

`examples/stage_b/contract_encoder_block.mlir` is not model-accurate: attention
is `q + k + v`, norm is affine scale/bias, activation is square, hidden size is
64, and parameters are operands. It tests shape, dispatch, forwarding, and
artifact accounting only.

## First Artifact

Generate a synthetic `.npz` with contract-shaped params and sample pre-embedded
inputs:

```sh
uv run --frozen --group numerical python scripts/make_stage_b_artifact.py \
  --output results/stage_b/encoder_block_synthetic.npz \
  --lengths 4,8,16,32,64,128
uv run --frozen --group numerical python scripts/check_stage_b_artifact.py \
  --input results/stage_b/encoder_block_synthetic.npz
```

`results/` is ignored. The file contains the ten parameter arrays, `x_sN`
sample inputs, and no length-specific parameter copies. It is benchmark
plumbing, not an exported model.

The next fixed adapter step packages those ten arrays into one canonical IREE
parameter archive and emits an MLIR module with symbolic module-level loads.
The public `@run_encoder_block` then accepts only the activation; the marked
internal `@encoder_block` retains explicit parameter operands for auditable
forwarding. See [stage-b-weight-bundle.md](stage-b-weight-bundle.md).

## IREE Benchmark

Run the artifact-backed Stage B benchmark:

```sh
uv run --frozen --group bench python benchmarks/bench_stage_b_iree.py \
  --artifact results/stage_b/encoder_block_synthetic.npz \
  --lengths 4,8,16,32,64,128
```

The benchmark compiles two IREE modules once: `dynamic_generic` and one
`dispatched_wrapper` containing all requested exact-length branches. It reports
artifact parameter bytes separately from VMFB bytes and checks wrapper output
against the dynamic generic output before timing.

Current snapshot, pinned to CPU 0:

```text
compiled modules:
variant              compile_s  vmfb_bytes
dynamic_generic         0.971       29764
dispatched_wrapper      3.396      102388

latency_ms:
S      dyn_med  wrap_med  speedup
4     0.060206  0.050735    1.187x
8     0.058857  0.053783    1.094x
16    0.075067  0.063446    1.183x
32    0.103553  0.082433    1.256x
64    0.161079  0.119304    1.350x
128   0.291003  0.204452    1.423x
```

Pinned IREE `20241104.1068` needs
`--iree-scheduling-optimize-bindings=false` for this wrapper shape.
