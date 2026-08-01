# Negative Results

## 2026-08-01: legacy IREE package does not reach E5's core boundary

The official f32 `intfloat/e5-small-v2` `model.onnx` at revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52` was SHA-validated, adapted to
batch one with in-graph pooling, checked against ONNX Runtime, and imported by
`iree-import-onnx` from pinned IREE `20241104.1068`. The imported Torch MLIR is
produced, but IREE input lowering fails before the requested
`global-optimization` textual boundary:

```sh
uv run --frozen --group e5 python scripts/build_e5_bridge.py \
  --mlir-opt build/llvm/bin/mlir-opt
```

```text
sentence_embedding.imported.mlir:202:12: error: failed to legalize operation
'torch.operator' that was explicitly marked illegal
%198 = torch.operator "onnx.Unsqueeze"(%arg1)
  {torch.onnx.axes = [1 : si64]}
  : (!torch.vtensor<[1,?],si64>) -> !torch.vtensor<[1,1,?],si64>
```

This `Unsqueeze` belongs to the canonical encoder's attention-mask path, not
the appended pooling operations. The legacy importer offers no
opset-selection option, so rewriting encoder operations was rejected.

Resolution: E5 uses the separately pinned `iree-base-compiler==3.11.0` and its
documented `--opset-version 17` import path. That frontend legalizes the
canonical graph and externalizes all 198 unchanged initializers directly. The
repository pins `iree-base-compiler==3.11.0` and
`iree-base-runtime==3.11.0`. Earlier Stage A/B results remain historical
measurements of IREE `20241104.1068`; they are not silently relabeled as
3.11.0 results.

Stage A and Stage B benchmark smoke tests and the Stage B external-parameter
numerical test still complete successfully with IREE `3.11.0`. Its Python
runtime bindings print `nanobind` leak diagnostics during interpreter shutdown;
the subprocess-based E5 baseline check avoids treating that upstream shutdown
noise as project output.

## 2026-07-04: Stage A S=16 MLIR Runner Loop Lowering

Command:

```sh
uv run --frozen python benchmarks/bench_mlp_s16.py --cpu 9
```

Ran with cgroups-v2/systemd isolation on logical CPUs `9,19`; the benchmark
pinned `mlir-runner` to CPU `9`.

Results:

```text
S=16 benchmark, warmup=5, iterations=25, repeats=10, affinity=cpu9
dynamic_generic: median_ms=23.254720 min_ms=23.129000 max_ms=23.329280 checksum=423.521
static_oracle: median_ms=23.254860 min_ms=23.160240 max_ms=25.540000 checksum=423.521
dispatched_wrapper: median_ms=23.306440 min_ms=23.131040 max_ms=23.362240 checksum=423.521
```

Conclusion: current MLIR runner loop lowering shows no measurable static-shape
win for the Stage A MLP at `S=16`.

Lowered IR inspection:

* `dynamic_generic.llvm.mlir`: 60,327 bytes, 27 branches, 32 calls, 10
  `malloc` calls, plus two descriptor-shape `llvm.alloca`s.
* `static_oracle.llvm.mlir`: 59,816 bytes, 27 branches, 32 calls, 10 `malloc`
  calls.
* The static oracle replaces sequence-size descriptor reads and dynamic
  allocation math with constants such as `16`, `24576`, and `6144`, but the
  scalar loop structure is otherwise effectively the same.

This only describes the current bufferization plus scalar loop runner path. It
does not rule out backend benefits from static shapes.

## 2026-07-05: Stage B IREE Binding Optimization

Pinned IREE `20241104.1068` mis-handles the multi-length Stage B wrapper when
`--iree-scheduling-optimize-bindings` is left enabled. The `4,8` wrapper fails
on `S=4` with a HAL constant-count mismatch. The benchmark disables that
optimization for both variants:

```text
--iree-scheduling-optimize-bindings=false
```

## 2026-07-04: IREE CPU Backend Snapshot

Command:

```sh
uv run --frozen --group bench python benchmarks/bench_iree.py \
  --lengths 16,1024,4096 \
  --dump-dir /tmp/shortseq-iree
```

IREE compiler/runtime: `20241104.1068`. Backend flags:
`--iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host`.

Results from this run, pinned to CPU 0 by the benchmark harness:

```text
S=16
dynamic_generic:    median_ms=0.299986 min_ms=0.293628 max_ms=0.307495 compile_s=0.550 vmfb_bytes=20700
static_oracle:      median_ms=0.349061 min_ms=0.342736 max_ms=0.352184 compile_s=0.548 vmfb_bytes=18596
dispatched_wrapper: median_ms=0.306506 min_ms=0.302693 max_ms=0.312045 compile_s=0.663 vmfb_bytes=32004

S=1024
dynamic_generic:    median_ms=5.744224 min_ms=5.537175 max_ms=5.962861 compile_s=0.544 vmfb_bytes=20700
static_oracle:      median_ms=5.676205 min_ms=5.493576 max_ms=5.891766 compile_s=0.528 vmfb_bytes=18532

S=4096
dynamic_generic:    median_ms=24.395650 min_ms=23.679063 max_ms=25.258560 compile_s=0.548 vmfb_bytes=20700
static_oracle:      median_ms=23.413159 min_ms=22.771027 max_ms=24.275000 compile_s=0.527 vmfb_bytes=18532
```

Takeaway: `S=16` still shows no static-oracle win and the dispatched artifact is
larger. `S=4096` shows a static-oracle win in this run, but that is not enough
to add a dispatch length without routing tests, numerical tests, and full
artifact accounting.
