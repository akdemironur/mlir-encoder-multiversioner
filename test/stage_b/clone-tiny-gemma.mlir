// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8,16})' \
// RUN:   %S/../../examples/stage_b/tiny_gemma_encoder.mlir | %FileCheck %s

// CHECK: module attributes {shortseq.ran} {
// CHECK-NOT: dense<

// CHECK-LABEL: func.func @encoder_generic(
// CHECK-SAME: tensor<1x?x64xf32>

// CHECK-LABEL: func.func private @encoder_s4(
// CHECK-SAME: tensor<1x4x64xf32>
// CHECK: -> tensor<1x4x64xf32>

// CHECK-LABEL: func.func private @encoder_s8(
// CHECK-SAME: tensor<1x8x64xf32>
// CHECK: -> tensor<1x8x64xf32>

// CHECK-LABEL: func.func private @encoder_s16(
// CHECK-SAME: tensor<1x16x64xf32>
// CHECK: -> tensor<1x16x64xf32>

// CHECK-LABEL: func.func @encoder(
// CHECK-SAME: %[[X:.*]]: tensor<1x?x64xf32>
// CHECK: %[[SEQ:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<1x?x64xf32>
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[IS4:.*]] = arith.cmpi eq, %[[SEQ]], %[[C4]] : index
// CHECK: scf.if %[[IS4]] -> (tensor<1x?x64xf32>) {
// CHECK: func.call @encoder_s4
// CHECK: } else {
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[IS8:.*]] = arith.cmpi eq, %[[SEQ]], %[[C8]] : index
// CHECK: scf.if %[[IS8]] -> (tensor<1x?x64xf32>) {
// CHECK: func.call @encoder_s8
// CHECK: } else {
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[IS16:.*]] = arith.cmpi eq, %[[SEQ]], %[[C16]] : index
// CHECK: scf.if %[[IS16]] -> (tensor<1x?x64xf32>) {
// CHECK: func.call @encoder_s16
// CHECK: } else {
// CHECK: func.call @encoder_generic
