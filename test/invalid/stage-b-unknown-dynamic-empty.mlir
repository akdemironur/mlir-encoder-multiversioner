// RUN: %not %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   %s 2>&1 | %FileCheck %s

module {
  func.func @encoder(
      %x: tensor<1x?x64xf32>,
      %q_w: tensor<64x64xf32>,
      %k_w: tensor<64x64xf32>,
      %v_w: tensor<64x64xf32>,
      %o_w: tensor<64x64xf32>,
      %norm_scale: tensor<64xf32>,
      %norm_bias: tensor<64xf32>,
      %ff_w1: tensor<64x256xf32>,
      %ff_b1: tensor<256xf32>,
      %ff_w2: tensor<256x64xf32>,
      %ff_b2: tensor<64xf32>
  ) -> tensor<1x?x64xf32> attributes {shortseq.entry, shortseq.stage_b} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x64xf32>
    %other = arith.addi %seq, %c1 : index
    %empty = tensor.empty(%other) : tensor<?x64xf32>
    %out = tensor.expand_shape %empty [[0, 1], [2]]
        output_shape [1, %seq, 64]
        : tensor<?x64xf32> into tensor<1x?x64xf32>
    return %out : tensor<1x?x64xf32>
  }
}

// CHECK: error: in @encoder, dynamic tensor.empty size must be tensor.dim of entry argument 0 at axis 1
