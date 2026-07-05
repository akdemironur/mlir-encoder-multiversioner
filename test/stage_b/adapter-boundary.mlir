// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8})' \
// RUN:   %S/../../examples/stage_b/preembed_adapter_boundary.mlir | %FileCheck %s

// CHECK-LABEL: func.func @stage_b_preembed_adapter(
// CHECK-SAME: tensor<1x?x64xf32>

// CHECK-LABEL: func.func @run_encoder(
// CHECK: call @stage_b_preembed_adapter
// CHECK: call @encoder(

// CHECK-LABEL: func.func @encoder_generic(
// CHECK-SAME: tensor<1x?x64xf32>

// CHECK-LABEL: func.func private @encoder_s4(
// CHECK-SAME: tensor<1x4x64xf32>

// CHECK-LABEL: func.func private @encoder_s8(
// CHECK-SAME: tensor<1x8x64xf32>

// CHECK-LABEL: func.func @encoder(
// CHECK-SAME: tensor<1x?x64xf32>
