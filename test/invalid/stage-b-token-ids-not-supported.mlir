// RUN: %not %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   %s 2>&1 | %FileCheck %s

module {
  func.func @encoder(
      %tokens: tensor<1x?xi32>,
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
    %seq = tensor.dim %tokens, %c1 : tensor<1x?xi32>
    %empty = tensor.empty(%seq) : tensor<1x?x64xf32>
    return %empty : tensor<1x?x64xf32>
  }
}

// CHECK: error: expected Stage B tiny encoder input argument 0 element type f32
