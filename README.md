# mlir-encoder-multiversioner

Out-of-tree MLIR pass playground for guarded static-shape specialization of
short-sequence encoder workloads. The project currently builds a loadable
`shortseq-specialize` pass plugin and a lit smoke test.

The repository intentionally uses a pinned source-built LLVM/MLIR tree, not the
Ubuntu MLIR development packages. The checkout and build products are local and
ignored by git; the pin is recorded in `toolchain/`.

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
uv run --frozen --group bench python test/mlp/mlp_numerical.py
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
uv run --frozen --group bench python test/stage_b/tiny_gemma_numerical.py
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
