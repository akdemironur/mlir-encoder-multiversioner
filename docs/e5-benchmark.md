# E5-small-v2 CPU benchmark

The Stage C benchmark compares three independently runnable configurations:

| Configuration | Source | Public entry |
| --- | --- | --- |
| `dynamic` | audited dynamic core | dynamic token widths |
| `static-oracle` | independently staticized ONNX frontend import | one fixed width |
| `dispatched` | four-variant multiversioned core | dynamic token widths |

The static oracle does not invoke `shortseq-specialize`. The model-specific
oracle builder replaces the already-audited ONNX `sequence_length` metadata
with one exact value, reruns the pinned ONNX frontend, and passes
`--no-save-params`. It verifies a fully static entry and the original 198
external references. All configurations therefore load the same `e5.irpa`;
the oracle builder creates no parameter archive.

Run `check-e5` first, then run the full benchmark:

```sh
cmake --build build/shortseq-pinned --target check-e5
uv run --frozen --group e5 python benchmarks/bench_e5_iree.py \
  --lengths 16,32,64,128 \
  --repeats 5 \
  --benchmark-min-time 0.2s \
  --warmup-seconds 0.5
```

Select one independently runnable configuration with `--variants dynamic`,
`--variants static-oracle`, or `--variants dispatched`. By default each child
compiler/runtime process is pinned to the first CPU in the current affinity
mask. `--cpu N` selects a different allowed CPU; `--no-affinity` disables
pinning.

Every run creates a new timestamped directory under
`results/e5-small-v2/benchmarks/` and refuses to overwrite a non-empty explicit
`--output-dir`. It retains the VMFBs, compiler output, IREE module size
breakdowns, per-repetition benchmark JSON, allocator statistics, warmed-RSS
probe output, numerical-check logs, fixtures, and `summary.json`.

## Measurement boundary and accounting

All VMFBs use exactly these pinned LLVM CPU flags:

```text
--iree-input-type=none
--iree-hal-target-device=local
--iree-hal-local-target-device-backends=llvm-cpu
--iree-llvmcpu-target-cpu=host
--iree-scheduling-optimize-bindings=false
```

The primary dispatched latency calls the public wrapper, so it includes all
three dimension reads, comparisons, conjunctions, and branch selection.
Google Benchmark performs the configured warmup before retaining repeated
trials. Tokenization, caller-side padding/bucketing, fixture file I/O, module
initialization, and parameter loading are outside the timed loop. Fixtures are
created before timing and are unpadded; all variants receive byte-identical
inputs and use `local-task`, one worker, and `parameter_mode=preload`.

Accounting is kept separate:

* canonical parameter bytes: `132852736` in one 198-entry IRPA;
* duplicated and length-specific parameter bytes: `0`;
* prepacked-weight bytes: not separately isolated; no second archive or learned
  payload is serialized, but that does not rule out runtime/compiler layouts;
* VMFB, VM bytecode, and embedded executable-rodata bytes: reported per module;
* variant metadata bytes: not separately isolated; functions, guards, and
  symbols remain included in the VM bytecode and VMFB counts;
* warmed process RSS: read from `/proc/self/status` in a separate pinned Python
  IREE-runtime process after independent warmup; it includes Python/binding
  overhead and uses a memory-mapped `ParameterIndex` provider, so it is a
  reproducible per-variant probe rather than the benchmark executable's RSS;
* active scratch bytes: reported as unavailable rather than guessed.

IREE 3.11.0's public allocator statistics combine scratch, inputs, outputs, and
other transient device allocations. The harness retains the exact
`DEVICE_LOCAL` peak as a strict scratch upper bound and also reports
`device_peak - canonical_parameter_bytes` as a literal arithmetic residual.
That residual may still contain prepacked or derived parameter layouts; it is
not presented as exact scratch or non-parameter memory.

## 2026-08-01 descriptive snapshot

This run used an Intel Core i9-10850K, logical CPU 0, IREE 3.11.0, five raw
repetitions, a 0.5-second warmup, and a 0.2-second minimum trial time. CPU
frequency scaling remained enabled. The raw evidence is retained locally at
`results/e5-small-v2/benchmarks/20260801T050332Z/`.

Artifact costs:

| Artifact | Frontend s | Compile s | VMFB bytes | VM bytecode | Embedded executable |
| --- | ---: | ---: | ---: | ---: | ---: |
| dynamic | — | 2.365 | 129796 | 57240 | 55284 |
| static oracle S=16 | 2.793 | 1.974 | 99598 | 30552 | 42600 |
| static oracle S=32 | 2.668 | 1.936 | 98222 | 30528 | 41224 |
| static oracle S=64 | 2.666 | 1.985 | 101454 | 30536 | 44456 |
| static oracle S=128 | 2.620 | 2.007 | 102063 | 30560 | 45072 |
| dispatched | — | 9.252 | 326903 | 138456 | 166508 |

Latency and memory distributions:

| S | Variant | Median ms | Min ms | Max ms | Warmed RSS bytes | DEVICE peak bytes | DEVICE peak − canonical |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | dynamic | 356.5847 | 356.2847 | 356.9789 | 138915840 | 133258752 | 406016 |
| 16 | static oracle | 353.7662 | 353.6970 | 354.1605 | 138801152 | 132877440 | 24704 |
| 16 | dispatched | 354.3658 | 354.2280 | 358.2756 | 139501568 | 133370880 | 518144 |
| 32 | dynamic | 710.5790 | 710.3001 | 711.9967 | 140361728 | 133689344 | 836608 |
| 32 | static oracle | 706.7536 | 706.4640 | 715.4303 | 139747328 | 132902144 | 49408 |
| 32 | dispatched | 684.7878 | 684.6923 | 685.2756 | 140414976 | 133518848 | 666112 |
| 64 | dynamic | 1430.5342 | 1424.8866 | 1432.8207 | 141864960 | 134624256 | 1771520 |
| 64 | static oracle | 1427.5026 | 1420.1287 | 1432.9957 | 142045184 | 132951552 | 98816 |
| 64 | dispatched | 1420.3898 | 1419.6607 | 1424.9518 | 142643200 | 133913088 | 1060352 |
| 128 | dynamic | 2916.6885 | 2838.2628 | 2923.8071 | 145620992 | 137182208 | 4329472 |
| 128 | static oracle | 2886.5994 | 2884.8136 | 2893.7097 | 148082688 | 135213568 | 2360832 |
| 128 | dispatched | 2892.0224 | 2872.6586 | 2898.5313 | 148140032 | 135782912 | 2930176 |

The final snapshot places both static-oracle and dispatched medians below the
dynamic median at all four lengths, but the relative gaps vary substantially;
at S=32 the dispatched and oracle distributions do not even overlap. Earlier
diagnostic reruns also changed the ordering at some lengths. CPU frequency
scaling remained enabled. No causal performance conclusion follows from these
wall-clock data. Explaining an effect would require supporting optimized IR,
generated code, or hardware-counter evidence, which is intentionally outside
this milestone.
