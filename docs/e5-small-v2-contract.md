# Stage C E5-small-v2 contract

Stage C expands the supported contract from pre-embedded synthetic inputs to
the official f32 `intfloat/e5-small-v2` ONNX encoder. The canonical artifact is
`model.onnx` at revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52`, with SHA-256
`4b8205be2a3c5fc53c6534d76a2012064f7309c162b806f2889c6ec8ec4fdcba`.
All acquisition URLs contain that immutable revision. Locally reproduced model
exports are not accepted.

## Public ABI and routing

```mlir
func.func @sentence_embedding(
    %input_ids: tensor<1x?xi64>,
    %attention_mask: tensor<1x?xi64>,
    %token_type_ids: tensor<1x?xi64>
) -> tensor<1x384xf32>
attributes {shortseq.entry, shortseq.e5_small_v2}
```

The initial exact lengths are `16,32,64,128`. A static branch is selected only
when the sequence dimension of all three operands equals that exact length:

```text
dim(input_ids, 1) == N &&
dim(attention_mask, 1) == N &&
dim(token_type_ids, 1) == N
```

Any otherwise-valid width tuple that does not match an exact length, including
mismatched widths, reaches the unmodified generic fallback. The pass never
pads, truncates, clamps, buckets, or rewrites IDs. Tokenization and optional
bucketing are caller responsibilities; if bucketing is used, the caller pads
all three tensors consistently. Callers also add the E5 `query: ` or
`passage: ` prefix before tokenization.

The runtime/test wrapper rejects masks containing values other than zero or
one, all-zero masks, token IDs outside `[0, 30522)`, and token types outside
`[0, 2)`. It also requires `1 <= S <= 512`, the model's position-embedding
range. Exact E5 specialization lengths above 512 are rejected by the pass. The
wrapper reports invalid input and does not modify it. The compiled graph does
not silently repair invalid input.

## Adapter and output

The adapter changes only the batch dimension of the three ONNX inputs to the
static value one. The sequence dimensions remain dynamic. It selects
`last_hidden_state`, expands the mask, and computes the model-card pooling:

```text
masked_sum = reduce_sum(last_hidden_state * float(mask)[..., none], axis=1)
count      = reduce_sum(float(mask), axis=1, keepdims=true)
mean       = masked_sum / count
embedding  = mean / sqrt(reduce_sum(mean * mean, axis=1, keepdims=true))
```

The sole output is f32 `[1,384]`. The adapter preserves every source
initializer name, dtype, shape, and serialized tensor bytes exactly.
All 197 learned initializers are f32. The only non-f32 initializer in the
canonical artifact is the non-learned i64 `[1,512]`
`embeddings.position_ids` metadata tensor; validation requires exactly this
exception rather than misclassifying it as a learned weight.

## Parameters and accounting

IREE must export one external archive with scope `e5`. Generic and exact
variants must reference that same archive; learned tensors must never appear as
dense payloads in clone bodies. Accounting reports these categories
separately:

* canonical IRPA parameter bytes;
* duplicated parameter bytes (required to remain zero);
* backend prepacked-weight bytes;
* VMFB executable-code bytes;
* variant metadata bytes;
* warmed process RSS and active scratch memory.

Every artifact or model-pool comparison must state whether weights, prepacked
weights, runtime contexts, and scratch buffers are shared or duplicated.
Length-specific prepacked-weight copies require a design note, benchmark, and
RSS accounting.

Token inputs and benchmark fixtures are not parameter bytes. Tokenization and
caller-side padding are outside latency. No model-pool memory comparison is
part of Stage C.

## Required coverage and benchmark impact

Positive routing and numerical coverage must include exact lengths
`16,32,64,128`, generic fallback length `17`, padded and unpadded masks, and
query/passage-prefixed fixtures. Rejection coverage includes wrong rank,
element type, batch, result type, missing token inputs, unsupported dynamic
axes, invalid ranges, and all-zero masks. Before enablement, each exact length
must have an IR routing test, numerical comparison, and benchmark
configuration.

The CPU benchmark keeps independently runnable dynamic, static-oracle, and
dispatched variants, warms each variant, retains repeated raw trials, pins CPU
affinity, and uses identical backend flags. It reports compilation time, VMFB
size, all parameter/prepacking/metadata categories, warmed RSS, scratch, and
latency distributions. Stage C makes no causal performance claim without
supporting IR, generated-code, or counter evidence. CUDA is a later milestone.

Model-dependent work belongs to `check-e5` and requires `fetch-e5` first.
`check-shortseq` remains network-independent.

The manually reviewed operation counts and the 20 allowed dynamic tensor
families are recorded in [e5-core-inventory.md](e5-core-inventory.md).
`check-e5` regenerates that inventory, transforms the full textual core,
verifies the result with upstream `mlir-opt`, and compiles the dispatched VMFB.

## Contract-expansion checklist

- Design note: this document.
- Positive tests: coordinated exact routing and full-model numerical checks.
- Negative tests: ABI/shape rejection and invalid runtime fixtures.
- README update: Stage C commands and boundary summary.
- Benchmark impact: four exact variants plus independent generic and oracle
  configurations, with the accounting above.
