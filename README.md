# mlir-encoder-multiversioner

Out-of-tree MLIR pass playground for guarded static-shape specialization of
short-sequence encoder workloads. The project currently builds a loadable
`shortseq-specialize` pass plugin and a lit smoke test.

The repository intentionally uses a pinned source-built LLVM/MLIR tree, not the
Ubuntu MLIR development packages. The checkout and build products are local and
ignored by git; the pin is recorded in `toolchain/`.

## Prerequisites

On Ubuntu, install the ordinary build tools plus clangd/clang for editor support
and for compiling this pass:

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
  FileCheck \
  not \
  count
```

This creates the tools and CMake package files used by the pass build:

```text
build/llvm/bin/mlir-opt
build/llvm/bin/FileCheck
build/llvm/bin/llvm-lit
build/llvm/lib/cmake/llvm
build/llvm/lib/cmake/mlir
```

## Build and test the pass

Configure the pass against the pinned LLVM/MLIR build:

```sh
cmake --preset pinned-llvm22
cmake --build --preset check-pinned-llvm22
```

The check target builds `build/shortseq-pinned/lib/ShortSeqPasses.so` and runs
the lit tests with the pinned `mlir-opt`.

To run the smoke test manually:

```sh
build/llvm/bin/mlir-opt \
  --load-pass-plugin=build/shortseq-pinned/lib/ShortSeqPasses.so \
  --pass-pipeline='builtin.module(shortseq-specialize)' \
  test/mlp/pass-invocation.mlir
```
