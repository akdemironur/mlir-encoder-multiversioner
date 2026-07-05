// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   %s | %FileCheck --check-prefix=S16 %s
// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8,16})' \
// RUN:   %s | %FileCheck --check-prefix=MULTI %s

module {
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    return %x : tensor<1x?x384xf32>
  }

  func.func @caller(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                    %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                    %b2: tensor<384xf32>) -> tensor<1x?x384xf32> {
    %y = func.call @mlp(%x, %w1, %b1, %w2, %b2)
        : (tensor<1x?x384xf32>, tensor<384x1536xf32>, tensor<1536xf32>,
           tensor<1536x384xf32>, tensor<384xf32>) -> tensor<1x?x384xf32>
    return %y : tensor<1x?x384xf32>
  }
}

// S16: module attributes {shortseq.ran} {
// S16-NOT: dense<
// S16-LABEL: func.func @mlp_generic(
// S16-SAME: tensor<1x?x384xf32>
// S16-LABEL: func.func private @mlp_s16(
// S16-SAME: tensor<1x16x384xf32>
// S16-SAME: tensor<384x1536xf32>
// S16-SAME: tensor<1536xf32>
// S16-SAME: tensor<1536x384xf32>
// S16-SAME: tensor<384xf32>
// S16-SAME: ) -> tensor<1x16x384xf32>
// S16-LABEL: func.func @mlp(
// S16-SAME: %[[X:.*]]: tensor<1x?x384xf32>
// S16-SAME: %[[W1:.*]]: tensor<384x1536xf32>
// S16-SAME: %[[B1:.*]]: tensor<1536xf32>
// S16-SAME: %[[W2:.*]]: tensor<1536x384xf32>
// S16-SAME: %[[B2:.*]]: tensor<384xf32>
// S16: %[[SEQ:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<1x?x384xf32>
// S16: %[[IS16:.*]] = arith.cmpi eq, %[[SEQ]], %{{.*}} : index
// S16: scf.if %[[IS16]] -> (tensor<1x?x384xf32>) {
// S16: %[[X16:.*]] = tensor.cast %[[X]] : tensor<1x?x384xf32> to tensor<1x16x384xf32>
// S16: func.call @mlp_s16(%[[X16]], %[[W1]], %[[B1]], %[[W2]], %[[B2]])
// S16: } else {
// S16: func.call @mlp_generic(%[[X]], %[[W1]], %[[B1]], %[[W2]], %[[B2]])
// S16-LABEL: func.func @caller(
// S16: call @mlp(
// S16-NOT: call @mlp_generic(

// MULTI-LABEL: func.func private @mlp_s4(
// MULTI-SAME: tensor<1x4x384xf32>
// MULTI-LABEL: func.func private @mlp_s8(
// MULTI-SAME: tensor<1x8x384xf32>
// MULTI-LABEL: func.func private @mlp_s16(
// MULTI-SAME: tensor<1x16x384xf32>
// MULTI-LABEL: func.func @mlp(
// MULTI-SAME: tensor<1x?x384xf32>
// MULTI: tensor.dim
// MULTI: arith.constant 4 : index
// MULTI: func.call @mlp_s4
// MULTI: arith.constant 8 : index
// MULTI: func.call @mlp_s8
// MULTI: arith.constant 16 : index
// MULTI: func.call @mlp_s16
// MULTI: func.call @mlp_generic
