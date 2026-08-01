// RUN: %mlir_opt --allow-unregistered-dialect \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8})' \
// RUN:   %S/../../examples/stage_b/external_parameter_adapter.mlir | %FileCheck %s

// CHECK: module attributes {shortseq.ran}
// CHECK-COUNT-10: "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::
// CHECK-NOT: dense<
// CHECK-LABEL: func.func @run_encoder_block(
// CHECK-COUNT-10: "util.global.load"()
// CHECK: call @encoder_block(
// CHECK-LABEL: func.func @encoder_block_generic(
// CHECK-LABEL: func.func private @encoder_block_s4(
// CHECK-SAME: tensor<1x4x64xf32>
// CHECK-LABEL: func.func private @encoder_block_s8(
// CHECK-SAME: tensor<1x8x64xf32>
// CHECK-LABEL: func.func @encoder_block(
// CHECK: func.call @encoder_block_s4
// CHECK: func.call @encoder_block_s8
// CHECK: func.call @encoder_block_generic
