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
    %slice = tensor.extract_slice %x[0, 0, 0] [1, %seq, 384] [1, 1, 1]
        : tensor<1x?x384xf32> to tensor<1x?x384xf32>
    return %slice : tensor<1x?x384xf32>
  }
}

// CHECK: error: in @mlp, unsupported dynamic tensor operation tensor.extract_slice
