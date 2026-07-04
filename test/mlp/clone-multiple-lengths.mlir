// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8,16})' \
// RUN:   %s | %FileCheck %s

module {
  func.func @mlp(
      %x: tensor<1x?x384xf32>,
      %w1: tensor<384x1536xf32>,
      %b1: tensor<1536xf32>,
      %w2: tensor<1536x384xf32>,
      %b2: tensor<384xf32>
  ) -> tensor<1x?x384xf32> attributes {shortseq.entry} {
    return %x : tensor<1x?x384xf32>
  }
}

// CHECK-LABEL: func.func @mlp_generic(

// CHECK-LABEL: func.func private @mlp_s4(
// CHECK-SAME: tensor<1x4x384xf32>
// CHECK-SAME: ) -> tensor<1x4x384xf32>

// CHECK-LABEL: func.func private @mlp_s8(
// CHECK-SAME: tensor<1x8x384xf32>
// CHECK-SAME: ) -> tensor<1x8x384xf32>

// CHECK-LABEL: func.func private @mlp_s16(
// CHECK-SAME: tensor<1x16x384xf32>
// CHECK-SAME: ) -> tensor<1x16x384xf32>

// CHECK-LABEL: func.func @mlp(
// CHECK-SAME: %[[X:.*]]: tensor<1x?x384xf32>
// CHECK: %[[SEQ:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<1x?x384xf32>
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[IS4:.*]] = arith.cmpi eq, %[[SEQ]], %[[C4]] : index
// CHECK: scf.if %[[IS4]] -> (tensor<1x?x384xf32>) {
// CHECK: func.call @mlp_s4
// CHECK: } else {
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[IS8:.*]] = arith.cmpi eq, %[[SEQ]], %[[C8]] : index
// CHECK: scf.if %[[IS8]] -> (tensor<1x?x384xf32>) {
// CHECK: func.call @mlp_s8
// CHECK: } else {
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[IS16:.*]] = arith.cmpi eq, %[[SEQ]], %[[C16]] : index
// CHECK: scf.if %[[IS16]] -> (tensor<1x?x384xf32>) {
// CHECK: func.call @mlp_s16
// CHECK: } else {
// CHECK: func.call @mlp_generic
