// RUN: %mlir_opt %S/../../examples/mlp/dynamic_mlp.mlir -verify-diagnostics -o /dev/null
// RUN: %mlir_opt %S/../../examples/mlp/dynamic_mlp.mlir --split-input-file -verify-diagnostics -o /dev/null
// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   %S/../../examples/mlp/dynamic_mlp.mlir -o /dev/null
// RUN: %mlir_opt %S/../../examples/mlp/static_mlp_s16.mlir -verify-diagnostics -o /dev/null
// RUN: %mlir_opt %S/../../examples/mlp/static_mlp_s16.mlir --split-input-file -verify-diagnostics -o /dev/null
