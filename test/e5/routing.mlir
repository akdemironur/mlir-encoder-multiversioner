// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=16,32,64,128})' \
// RUN:   %s | %FileCheck %s

// The E5 branch is taken only when all three token tensors have the same
// exact configured width. The fallback receives the original values.
module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// CHECK: module attributes {shortseq.ran} {
// CHECK-NOT: dense<
// CHECK-LABEL: func.func @sentence_embedding_generic(
// CHECK-SAME: tensor<1x?xi64>
// CHECK-SAME: tensor<1x?xi64>
// CHECK-SAME: tensor<1x?xi64>
// CHECK-SAME: ) -> tensor<1x384xf32>
// CHECK-NOT: shortseq.e5_small_v2
// CHECK-LABEL: func.func private @sentence_embedding_s16(
// CHECK-SAME: tensor<1x16xi64>
// CHECK-SAME: tensor<1x16xi64>
// CHECK-SAME: tensor<1x16xi64>
// CHECK-SAME: ) -> tensor<1x384xf32>
// CHECK-LABEL: func.func private @sentence_embedding_s32(
// CHECK-SAME: tensor<1x32xi64>
// CHECK-SAME: tensor<1x32xi64>
// CHECK-SAME: tensor<1x32xi64>
// CHECK-LABEL: func.func private @sentence_embedding_s64(
// CHECK-SAME: tensor<1x64xi64>
// CHECK-SAME: tensor<1x64xi64>
// CHECK-SAME: tensor<1x64xi64>
// CHECK-LABEL: func.func private @sentence_embedding_s128(
// CHECK-SAME: tensor<1x128xi64>
// CHECK-SAME: tensor<1x128xi64>
// CHECK-SAME: tensor<1x128xi64>
// CHECK-LABEL: func.func @sentence_embedding(
// CHECK-SAME: %[[IDS:[a-zA-Z0-9_]+]]: tensor<1x?xi64>
// CHECK-SAME: %[[MASK:[a-zA-Z0-9_]+]]: tensor<1x?xi64>
// CHECK-SAME: %[[TYPES:[a-zA-Z0-9_]+]]: tensor<1x?xi64>
// CHECK-SAME: ) -> tensor<1x384xf32>
// CHECK: %[[IDS_DIM:.*]] = tensor.dim %[[IDS]], %{{.*}} : tensor<1x?xi64>
// CHECK: %[[MASK_DIM:.*]] = tensor.dim %[[MASK]], %{{.*}} : tensor<1x?xi64>
// CHECK: %[[TYPES_DIM:.*]] = tensor.dim %[[TYPES]], %{{.*}} : tensor<1x?xi64>
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[IDS_16:.*]] = arith.cmpi eq, %[[IDS_DIM]], %[[C16]] : index
// CHECK: %[[MASK_16:.*]] = arith.cmpi eq, %[[MASK_DIM]], %[[C16]] : index
// CHECK: %[[BOTH_16:.*]] = arith.andi %[[IDS_16]], %[[MASK_16]] : i1
// CHECK: %[[TYPES_16:.*]] = arith.cmpi eq, %[[TYPES_DIM]], %[[C16]] : index
// CHECK: %[[ALL_16:.*]] = arith.andi %[[BOTH_16]], %[[TYPES_16]] : i1
// CHECK: scf.if %[[ALL_16]] -> (tensor<1x384xf32>) {
// CHECK: %[[IDS_CAST:.*]] = tensor.cast %[[IDS]] : tensor<1x?xi64> to tensor<1x16xi64>
// CHECK: %[[MASK_CAST:.*]] = tensor.cast %[[MASK]] : tensor<1x?xi64> to tensor<1x16xi64>
// CHECK: %[[TYPES_CAST:.*]] = tensor.cast %[[TYPES]] : tensor<1x?xi64> to tensor<1x16xi64>
// CHECK: func.call @sentence_embedding_s16(%[[IDS_CAST]], %[[MASK_CAST]], %[[TYPES_CAST]])
// CHECK: func.call @sentence_embedding_s32
// CHECK: func.call @sentence_embedding_s64
// CHECK: func.call @sentence_embedding_s128
// CHECK: func.call @sentence_embedding_generic(%[[IDS]], %[[MASK]], %[[TYPES]])
