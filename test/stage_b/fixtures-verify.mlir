// RUN: %mlir_opt %S/../../examples/stage_b/tiny_gemma_encoder.mlir -verify-diagnostics -o /dev/null
// RUN: %mlir_opt %S/../../examples/stage_b/tiny_gemma_encoder.mlir --split-input-file -verify-diagnostics -o /dev/null
// RUN: %mlir_opt %S/../../examples/stage_b/preembed_adapter_boundary.mlir -verify-diagnostics -o /dev/null
