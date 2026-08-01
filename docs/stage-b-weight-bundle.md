# Stage B Weight Bundle

## Decision

Stage B uses one external IREE parameter archive (`.irpa`) plus an MLIR module
containing symbolic parameter references. Weight bytes are not emitted as
`dense<...>` attributes and are not arguments of the public runtime entry.

The marked `@encoder_block` keeps its explicit flat parameter operands. This is
an internal compiler ABI: `shortseq-specialize` can refine activation shapes and
forward the same SSA parameter values to every exact-length variant and the
generic fallback. An unmarked `@run_encoder_block` loads each canonical
parameter once from a module-level `util.global` reference and calls the marked
entry.

A heterogeneous MLIR tuple is not used as the compiler ABI. It would require
container construction/extraction operations, obscure operand-level sharing
checks, and add no storage sharing beyond the module-level globals. Python or C++
application code may expose a bundle object for ergonomics; that object maps to
the fixed flat internal operand order.

## Invariant

For each source parameter name there is exactly:

* one key in the `.irpa` archive;
* one `#flow.parameter.named<scope::key>` module-level reference;
* one load in the unmarked public adapter;
* no parameter payload in `@encoder_block`, `@encoder_block_generic`, or any
  `@encoder_block_sN` clone.

The external scope defaults to `stage_b`. Runtime binding must use the same
scope. The archive contains opaque aligned bytes; shapes and dtypes remain part
of the MLIR contract and are checked when the bundle is built.

## Why Not Embed Dense Constants in the Marked Function?

The pass clones the marked function body. A dense weight constant in that body
would therefore appear in every static clone as well as the generic body. Even
if a later compiler happened to deduplicate the bytes, the pre-IREE IR would
violate the canonical-representation invariant and make artifact accounting
backend-dependent.

Putting constants only in an unmarked outer function avoids pass-time cloning,
but still inflates textual MLIR and VMFBs. Symbolic globals plus an IRPA archive
keep code and canonical weight storage separate and let the IREE runtime map or
preload the archive.

## Fixed Export Flow

Generate the existing synthetic artifact, then build and check the bundle:

```sh
uv run --frozen --group numerical python scripts/make_stage_b_artifact.py \
  --output results/stage_b/encoder_block_synthetic.npz \
  --lengths 4,8
uv run --frozen --group bench python scripts/make_stage_b_bundle.py \
  --artifact results/stage_b/encoder_block_synthetic.npz
uv run --frozen --group bench python scripts/check_stage_b_bundle.py
uv run --frozen --group bench python \
  test/stage_b/external_parameter_bundle_numerical.py
```

Run the specialization pass with the upstream toolchain. The generated MLIR
uses generic spelling for IREE operations so they can be preserved without
registering IREE dialects in the pass plugin:

```sh
build/llvm/bin/mlir-opt \
  --allow-unregistered-dialect \
  --load-pass-plugin=build/shortseq-pinned/lib/ShortSeqPasses.so \
  --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8})' \
  results/stage_b/encoder_block_bundle.mlir \
  -o results/stage_b/encoder_block_multiversioned.mlir
```

Compile and run with the pinned IREE tools:

```sh
uv run --frozen --group bench iree-compile \
  results/stage_b/encoder_block_multiversioned.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  --iree-scheduling-optimize-bindings=false \
  -o results/stage_b/encoder_block.vmfb
uv run --frozen --group bench iree-run-module \
  --module=results/stage_b/encoder_block.vmfb \
  --parameters=stage_b=results/stage_b/encoder_block_weights.irpa \
  --function=run_encoder_block \
  --input=1x4x64xf32=0.1
```

The two optimization tools are intentionally separate. The repository pass
plugin is built against the pinned upstream MLIR tree and is not ABI-compatible
with the MLIR library bundled inside the IREE Python wheel.

## Accounting and Benchmark Impact

The synthetic ten-parameter archive has 198,400 canonical parameter bytes and
zero length-specific parameter bytes. `check_stage_b_bundle.py` validates both
the symbolic references and archive byte ranges. Prepacked-weight bytes, VMFB
code bytes, variant metadata, warmed RSS, and active scratch remain separate
measurements; the bundle builder does not infer them.

External loading adds module initialization and parameter-provider costs. No
latency or memory improvement is claimed until the artifact-backed benchmark
measures the operand and external-archive paths with identical compiler flags,
warmup, and runtime parameter mode.
