// RUN: %not %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   %s 2>&1 | %FileCheck %s

module {
  func.func @mlp(
      %x: tensor<1x?x384xf32>,
      %w1: tensor<384x1536xf32>,
      %b1: tensor<1536xf32>,
      %w2: tensor<1536x384xf32>,
      %b2: tensor<384xf32>
  ) -> tensor<1x?x384xf32> attributes {shortseq.entry} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x384xf32>
    %other = arith.addi %seq, %c1 : index
    %empty = tensor.empty(%other) : tensor<?x384xf32>
    %y = tensor.expand_shape %empty [[0, 1], [2]]
        output_shape [1, %seq, 384]
        : tensor<?x384xf32> into tensor<1x?x384xf32>
    return %y : tensor<1x?x384xf32>
  }
}

// CHECK: error: in @mlp, dynamic tensor.empty size must be tensor.dim of entry argument 0 at axis 1
