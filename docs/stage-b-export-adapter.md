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
  --lengths 4,8,16
```

`results/` is ignored. The file contains the ten parameter arrays, `x_sN`
sample inputs, and metadata. It is benchmark plumbing, not an exported model.
