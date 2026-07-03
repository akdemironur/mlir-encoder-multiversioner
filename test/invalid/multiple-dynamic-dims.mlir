// RUN: %not %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   %s 2>&1 | %FileCheck %s

module {
  func.func @mlp(
      %x: tensor<1x?x?xf32>,
      %w1: tensor<384x1536xf32>,
      %b1: tensor<1536xf32>,
      %w2: tensor<1536x384xf32>,
      %b2: tensor<384xf32>
  ) -> tensor<1x?x?xf32> attributes {shortseq.entry} {
    return %x : tensor<1x?x?xf32>
  }
}

// CHECK: error: shortseq-specialize requires exactly one dynamic tensor dimension on the entry input
