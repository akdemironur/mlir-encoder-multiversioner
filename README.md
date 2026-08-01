# mlir-encoder-multiversioner

Out-of-tree MLIR pass for guarded static-shape specialization of
short-sequence encoder inference. The project builds a loadable
`shortseq-specialize` pass plugin, IR tests, numerical checks, artifact
accounting, and IREE CPU benchmarks.

The repository intentionally uses a pinned source-built LLVM/MLIR tree, not the
Ubuntu MLIR development packages. The checkout and build products are local and
ignored by git; the pin is recorded in `toolchain/`.

## What It Does

`shortseq-specialize` takes one marked dynamic sequence-length entry point,
clones it into exact-length static variants, and keeps the original behavior as
a generic fallback.

The optimization hypothesis is simple: if sequence length is static, downstream
lowering should be able to remove shape operations, simplify allocation sizes,
and make better loop/vectorization decisions. This pass performs the guarded
clone, type refinement, legality checks, and parameter forwarding. The actual
speedup depends on the downstream compiler/backend exploiting the static IR.

## IR Shape

Input:

```mlir
func.func @mlp(
    %x: tensor<1x?x384xf32>,
    %w1: tensor<384x1536xf32>,
    %b1: tensor<1536xf32>,
    %w2: tensor<1536x384xf32>,
    %b2: tensor<384xf32>
) -> tensor<1x?x384xf32> attributes {shortseq.entry} {
  ...
}
```

Generated static clone:

```mlir
func.func private @mlp_s16(
    %x: tensor<1x16x384xf32>,
    %w1: tensor<384x1536xf32>,
    %b1: tensor<1536xf32>,
    %w2: tensor<1536x384xf32>,
    %b2: tensor<384xf32>
) -> tensor<1x16x384xf32> {
  ...
  %hidden = tensor.empty() : tensor<16x1536xf32>
  ...
}
```

Generated wrapper:

```mlir
func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
               %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
               %b2: tensor<384xf32>) -> tensor<1x?x384xf32> {
  %c1 = arith.constant 1 : index
  %s = tensor.dim %x, %c1 : tensor<1x?x384xf32>
  %c16 = arith.constant 16 : index
  %is16 = arith.cmpi eq, %s, %c16 : index
  %0 = scf.if %is16 -> (tensor<1x?x384xf32>) {
    %xs = tensor.cast %x : tensor<1x?x384xf32> to tensor<1x16x384xf32>
    %ys = func.call @mlp_s16(%xs, %w1, %b1, %w2, %b2)
      : (tensor<1x16x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>,
         tensor<1536x384xf32>, tensor<384xf32>) -> tensor<1x16x384xf32>
    %y = tensor.cast %ys : tensor<1x16x384xf32> to tensor<1x?x384xf32>
    scf.yield %y : tensor<1x?x384xf32>
  } else {
    %y = func.call @mlp_generic(%x, %w1, %b1, %w2, %b2)
      : (tensor<1x?x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>,
         tensor<1536x384xf32>, tensor<384xf32>) -> tensor<1x?x384xf32>
    scf.yield %y : tensor<1x?x384xf32>
  }
  return %0 : tensor<1x?x384xf32>
}
```

Parameter operands are forwarded. Learned-weight payloads are not cloned into
specialized functions.

## Results

Stage B uses a synthetic encoder-block-shaped fixture with pre-embedded
`tensor<1x?x64xf32>` input and ten f32 parameter operands. It is not a real
model export yet.

Environment:

```text
machine: Intel Core i9-10850K, CPU 0 pinned
LLVM/MLIR: llvmorg-22.1.8, llvm-project ca7933e47d3a3451d81e72ac174dcb5aa28b59d1
IREE: 20241104.1068, llvm-cpu
flags: --iree-llvmcpu-target-cpu=host --iree-scheduling-optimize-bindings=false
warmup: 10
iterations: 50
repeats: 10
lengths: 4,8,16,32,64,128
```

Compiled artifacts:

```text
variant              compile_s  vmfb_bytes
dynamic_generic         0.971       29764
dispatched_wrapper      3.396      102388
```

Latency, median and max over repeats:

```text
S      dyn_med  dyn_max  wrap_med  wrap_max  speedup
4     0.060206 0.082878  0.050735  0.081711   1.187x
8     0.058857 0.086316  0.053783  0.062525   1.094x
16    0.075067 0.101681  0.063446  0.074958   1.183x
32    0.103553 0.149356  0.082433  0.097929   1.256x
64    0.161079 0.196392  0.119304  0.166256   1.350x
128   0.291003 0.322170  0.204452  0.230617   1.423x
```

Conclusion: this synthetic Stage B fixture shows a real IREE CPU speedup from
static sequence-length dispatch, at the cost of larger executable size and
longer compile time. The result does not yet claim speedup for a real exported
encoder.

## Prerequisites

On Ubuntu, install the ordinary build tools, `uv` for the Python harness, and
clangd/clang for editor support and for compiling this pass:

```sh
sudo apt-get install \
  git cmake ninja-build python3 g++ ccache \
  clang-22 clang++-22 clangd-22 \
  zlib1g-dev libzstd-dev
```

## Build the pinned LLVM/MLIR tools

Fetch the pinned LLVM source:

```sh
mkdir -p third_party
git clone https://github.com/llvm/llvm-project.git third_party/llvm-project
git -C third_party/llvm-project checkout "$(cat toolchain/llvm-project.commit)"
```

Configure and build the small LLVM/MLIR toolchain needed by this project:

```sh
cmake -G Ninja \
  -S third_party/llvm-project/llvm \
  -B build/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INCLUDE_TESTS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/clang-22 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-22 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

ninja -C build/llvm -j8 \
  mlir-opt \
  mlir-runner \
  FileCheck \
  not \
  count \
  mlir_c_runner_utils \
  mlir_runner_utils
```

This creates the tools and CMake package files used by the pass build:

```text
build/llvm/bin/mlir-opt
build/llvm/bin/mlir-runner
build/llvm/bin/FileCheck
build/llvm/bin/llvm-lit
build/llvm/lib/cmake/llvm
build/llvm/lib/cmake/mlir
build/llvm/lib/libmlir_c_runner_utils.so
build/llvm/lib/libmlir_runner_utils.so
```

## Build and test the pass

Configure the pass against the pinned LLVM/MLIR build:

```sh
cmake --preset pinned-llvm22
cmake --build --preset check-pinned-llvm22
```

The check target builds `build/shortseq-pinned/lib/ShortSeqPasses.so`, runs the
lit tests with the pinned `mlir-opt`, and runs the Stage A/Stage B accounting
and numerical checks.

To run only the numerical harness:

```sh
uv run --frozen --group numerical python test/mlp/mlp_numerical.py
```

To run only the parameter/artifact accounting check:

```sh
uv run --frozen python scripts/check_parameter_accounting.py \
  --mlir-opt build/llvm/bin/mlir-opt \
  --plugin build/shortseq-pinned/lib/ShortSeqPasses.so \
  --input examples/mlp/dynamic_mlp.mlir \
  --lengths 4,8,16
```

For the Stage B fixture, use:

```sh
uv run --frozen python scripts/check_parameter_accounting.py \
  --mlir-opt build/llvm/bin/mlir-opt \
  --plugin build/shortseq-pinned/lib/ShortSeqPasses.so \
  --input examples/stage_b/tiny_gemma_encoder.mlir \
  --entry encoder \
  --lengths 4,8,16
uv run --frozen --group numerical python test/stage_b/tiny_gemma_numerical.py
```

To run the initial S=16 MLIR runner benchmark:

```sh
uv run --frozen python benchmarks/bench_mlp_s16.py
```

By default the benchmark runs `5` warmup invocations, `25` timed invocations per
repeat, and `10` repeats. It pins each `mlir-runner` process to the first CPU in
the current affinity mask; use `--cpu N` to select a core, `--no-affinity` to
disable pinning. Each variant reports MLIR size and simple lowered-IR counts
for dims, branches, calls, allocs, and frees. Use `--dump-dir DIR` to keep
generated/lowered MLIR; add `--dump-objects` to also keep runner objects and
report `object_bytes`.

Use `--pipeline affine` for the experimental affine-loop inspection path. Check
`static_oracle` first; dispatch only matters after the oracle changes:

```sh
uv run --frozen python benchmarks/bench_mlp_s16.py \
  --pipeline affine \
  --dump-dir /tmp/shortseq-s16-affine
```

To run the IREE CPU benchmark for selected sequence lengths:

```sh
uv run --frozen --group bench python benchmarks/bench_iree.py \
  --lengths 16,1024,4096 \
  --dump-dir /tmp/shortseq-iree
```

On the current i9-10850K topology, logical CPUs `9` and `19` are SMT siblings
on physical core `9`. On a cgroups v2 system managed by systemd, move the
normal userspace slices off that sibling pair, run the benchmark in its own
slice, and pin the measured `mlir-runner` process to logical CPU `9`:

```sh
UV_BIN="$(command -v uv)"

restore_cpus() {
  for unit in system.slice user.slice init.scope; do
    sudo systemctl set-property --runtime "$unit" AllowedCPUs=0-19
  done
}
trap restore_cpus EXIT

for unit in system.slice user.slice init.scope; do
  sudo systemctl set-property --runtime "$unit" AllowedCPUs=0-8,10-18
done

sudo systemd-run --scope --collect --same-dir \
  --slice=benchmark.slice \
  --uid="$(id -u)" --gid="$(id -g)" \
  -p AllowedCPUs=9,19 \
  -E UV_CACHE_DIR=/tmp/shortseq-uv-cache \
  "$UV_BIN" run --frozen python benchmarks/bench_mlp_s16.py --cpu 9

restore_cpus
trap - EXIT
```

This is runtime cgroup isolation for systemd-managed userspace. It does not
fully isolate kernel work, IRQs, or firmware effects; use boot-time CPU
isolation and IRQ affinity for stricter benchmark runs.

To run the smoke test manually:

```sh
build/llvm/bin/mlir-opt \
  --load-pass-plugin=build/shortseq-pinned/lib/ShortSeqPasses.so \
  --pass-pipeline='builtin.module(shortseq-specialize)' \
  test/mlp/routing.mlir
```

To emit several static variants and a dynamic fallback wrapper:

```sh
build/llvm/bin/mlir-opt \
  --load-pass-plugin=build/shortseq-pinned/lib/ShortSeqPasses.so \
  --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8,16})' \
  examples/mlp/dynamic_mlp.mlir
```

There is also an experimental Stage B tiny encoder fixture. It uses
pre-embedded `tensor<1x?x64xf32>` input with synthetic parameter operands; token
ids, embedding lookup, masks, and real model artifacts are still out of scope.
The exact contract is documented in [stage-b-contract.md](docs/stage-b-contract.md).
The adapter boundary example keeps that split explicit: unmarked adapter code
produces pre-embedded f32 activations, and only `@encoder` is specialized.

```sh
build/llvm/bin/mlir-opt \
  --load-pass-plugin=build/shortseq-pinned/lib/ShortSeqPasses.so \
  --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8,16})' \
  examples/stage_b/tiny_gemma_encoder.mlir

build/llvm/bin/mlir-opt \
  --load-pass-plugin=build/shortseq-pinned/lib/ShortSeqPasses.so \
  --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8})' \
  examples/stage_b/preembed_adapter_boundary.mlir

build/llvm/bin/mlir-opt \
  --load-pass-plugin=build/shortseq-pinned/lib/ShortSeqPasses.so \
  --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8})' \
  examples/stage_b/contract_encoder_block.mlir
```

For the fixed external-weight adapter, build one `.irpa` archive and a symbolic
MLIR module whose public entry takes only the activation. Weight operands remain
an internal ABI so every static variant and the generic fallback receive the
same canonical values. See
[stage-b-weight-bundle.md](docs/stage-b-weight-bundle.md) for commands and
accounting boundaries.

## Stage C: E5-small-v2 token-to-embedding

Stage C targets the official f32 `intfloat/e5-small-v2` ONNX artifact at the
immutable revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`. Its public
entry accepts batch-1 dynamic-width `input_ids`, `attention_mask`, and
`token_type_ids` tensors and returns one mask-pooled, L2-normalized
`tensor<1x384xf32>` embedding. Exact lengths are `16,32,64,128`; all three
widths must equal the selected length. Otherwise-valid width tuples use the
generic fallback, while invalid fixtures are rejected by the caller-side
wrapper. The pass never performs caller-side padding or tokenization.

The full ABI, input validation, parameter sharing, accounting boundaries, and
benchmark checklist are in
[docs/e5-small-v2-contract.md](docs/e5-small-v2-contract.md).

Model files are ignored build inputs. Fetch and validate the pinned artifacts:

```sh
cmake --build build/shortseq-pinned --target fetch-e5
cmake --build build/shortseq-pinned --target check-e5
```

Equivalently, the model-specific acquisition command is:

```sh
uv run --frozen python scripts/fetch_e5_artifacts.py
```

`check-shortseq` does not fetch or inspect E5 files and remains
network-independent. `check-e5` enforces a hard bridge gate: IREE's pinned ONNX
importer must reach a textual input/global-optimization boundary, export one
`e5`-scoped IRPA, round-trip through upstream `mlir-opt`, and resume as an
independently compiled dynamic baseline. Pass legality must remain unchanged
until that gate succeeds.

The current toolchain pins IREE `3.11.0` for both existing benchmarks and the
E5 bridge. Earlier Stage A/B result tables remain historical measurements from
IREE `20241104.1068`; rerun them before drawing comparisons with Stage C.
