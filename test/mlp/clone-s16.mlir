// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
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

  func.func @caller(
      %x: tensor<1x?x384xf32>,
      %w1: tensor<384x1536xf32>,
      %b1: tensor<1536xf32>,
      %w2: tensor<1536x384xf32>,
      %b2: tensor<384xf32>
  ) -> tensor<1x?x384xf32> {
    %y = func.call @mlp(%x, %w1, %b1, %w2, %b2)
        : (tensor<1x?x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>,
           tensor<1536x384xf32>, tensor<384xf32>) -> tensor<1x?x384xf32>
    return %y : tensor<1x?x384xf32>
  }
}

// CHECK: module attributes {shortseq.ran} {
// CHECK-NOT: dense<
// CHECK-LABEL: func.func @mlp_generic(
// CHECK-SAME: %[[X:.*]]: tensor<1x?x384xf32>
// CHECK-SAME: %[[W1:.*]]: tensor<384x1536xf32>
// CHECK-SAME: %[[B1:.*]]: tensor<1536xf32>
// CHECK-SAME: %[[W2:.*]]: tensor<1536x384xf32>
// CHECK-SAME: %[[B2:.*]]: tensor<384xf32>
// CHECK-SAME: ) -> tensor<1x?x384xf32>
// CHECK: return %[[X]] : tensor<1x?x384xf32>
// CHECK-LABEL: func.func private @mlp_s16(
// CHECK-SAME: tensor<1x16x384xf32>
// CHECK-SAME: tensor<384x1536xf32>
// CHECK-SAME: tensor<1536xf32>
// CHECK-SAME: tensor<1536x384xf32>
// CHECK-SAME: tensor<384xf32>
// CHECK-SAME: ) -> tensor<1x16x384xf32>
// CHECK-LABEL: func.func @caller(
// CHECK: call @mlp_generic(
// CHECK-NOT: dense<
