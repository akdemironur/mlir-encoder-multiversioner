 // RUN: %mlir_opt %S/../../examples/mlp/dynamic_mlp.mlir -verify-diagnostics -o /dev/null
 // RUN: %mlir_opt %S/../../examples/mlp/dynamic_mlp.mlir --split-input-file -verify-diagnostics -o /dev/null
 // RUN: %mlir_opt %S/../../examples/mlp/static_mlp_s16.mlir -verify-diagnostics -o /dev/null
 // RUN: %mlir_opt %S/../../examples/mlp/static_mlp_s16.mlir --split-input-file -verify-diagnostics -o /dev/null
