# Negative Results

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
