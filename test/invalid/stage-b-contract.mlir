// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   --split-input-file -verify-diagnostics %s -o /dev/null

module {
  // expected-error @+1 {{expected Stage B tiny encoder entry to have 11 arguments}}
  func.func @encoder(%x: tensor<1x?x64xf32>, %mask: tensor<1x?xf32>,
                     %q_w: tensor<64x64xf32>, %k_w: tensor<64x64xf32>,
                     %v_w: tensor<64x64xf32>, %o_w: tensor<64x64xf32>,
                     %norm_scale: tensor<64xf32>, %norm_bias: tensor<64xf32>,
                     %ff_w1: tensor<64x256xf32>, %ff_b1: tensor<256xf32>,
                     %ff_w2: tensor<256x64xf32>, %ff_b2: tensor<64xf32>)
      -> tensor<1x?x64xf32> attributes {shortseq.entry, shortseq.stage_b} {
    return %x : tensor<1x?x64xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected Stage B tiny encoder entry to have 1 result}}
  func.func @encoder(%x: tensor<1x?x64xf32>, %q_w: tensor<64x64xf32>,
                     %k_w: tensor<64x64xf32>, %v_w: tensor<64x64xf32>,
                     %o_w: tensor<64x64xf32>, %norm_scale: tensor<64xf32>,
                     %norm_bias: tensor<64xf32>, %ff_w1: tensor<64x256xf32>,
                     %ff_b1: tensor<256xf32>, %ff_w2: tensor<256x64xf32>,
                     %ff_b2: tensor<64xf32>)
      -> (tensor<1x?x64xf32>, tensor<1x?x64xf32>)
      attributes {shortseq.entry, shortseq.stage_b} {
    return %x, %x : tensor<1x?x64xf32>, tensor<1x?x64xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected Stage B tiny encoder input argument 0 element type f32}}
  func.func @encoder(%tokens: tensor<1x?xi32>, %q_w: tensor<64x64xf32>,
                     %k_w: tensor<64x64xf32>, %v_w: tensor<64x64xf32>,
                     %o_w: tensor<64x64xf32>, %norm_scale: tensor<64xf32>,
                     %norm_bias: tensor<64xf32>, %ff_w1: tensor<64x256xf32>,
                     %ff_b1: tensor<256xf32>, %ff_w2: tensor<256x64xf32>,
                     %ff_b2: tensor<64xf32>)
      -> tensor<1x?x64xf32> attributes {shortseq.entry, shortseq.stage_b} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %tokens, %c1 : tensor<1x?xi32>
    %empty = tensor.empty(%seq) : tensor<1x?x64xf32>
    return %empty : tensor<1x?x64xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected hidden dimension 64, got 128}}
  func.func @encoder(%x: tensor<1x?x128xf32>, %q_w: tensor<64x64xf32>,
                     %k_w: tensor<64x64xf32>, %v_w: tensor<64x64xf32>,
                     %o_w: tensor<64x64xf32>, %norm_scale: tensor<64xf32>,
                     %norm_bias: tensor<64xf32>, %ff_w1: tensor<64x256xf32>,
                     %ff_b1: tensor<256xf32>, %ff_w2: tensor<256x64xf32>,
                     %ff_b2: tensor<64xf32>)
      -> tensor<1x?x64xf32> attributes {shortseq.entry, shortseq.stage_b} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x128xf32>
    %empty = tensor.empty(%seq) : tensor<1x?x64xf32>
    return %empty : tensor<1x?x64xf32>
  }
}

// -----

module {
  func.func @encoder(%x: tensor<1x?x64xf32>, %q_w: tensor<64x64xf32>,
                     %k_w: tensor<64x64xf32>, %v_w: tensor<64x64xf32>,
                     %o_w: tensor<64x64xf32>, %norm_scale: tensor<64xf32>,
                     %norm_bias: tensor<64xf32>, %ff_w1: tensor<64x256xf32>,
                     %ff_b1: tensor<256xf32>, %ff_w2: tensor<256x64xf32>,
                     %ff_b2: tensor<64xf32>)
      -> tensor<1x?x64xf32> attributes {shortseq.entry, shortseq.stage_b} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x64xf32>
    // expected-error @+1 {{unsupported dynamic tensor operation tensor.extract_slice}}
    %slice = tensor.extract_slice %x[0, 0, 0] [1, %seq, 64] [1, 1, 1]
        : tensor<1x?x64xf32> to tensor<1x?x64xf32>
    return %slice : tensor<1x?x64xf32>
  }
}

// -----

module {
  func.func @encoder(%x: tensor<1x?x64xf32>, %q_w: tensor<64x64xf32>,
                     %k_w: tensor<64x64xf32>, %v_w: tensor<64x64xf32>,
                     %o_w: tensor<64x64xf32>, %norm_scale: tensor<64xf32>,
                     %norm_bias: tensor<64xf32>, %ff_w1: tensor<64x256xf32>,
                     %ff_b1: tensor<256xf32>, %ff_w2: tensor<256x64xf32>,
                     %ff_b2: tensor<64xf32>)
      -> tensor<1x?x64xf32> attributes {shortseq.entry, shortseq.stage_b} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x64xf32>
    %other = arith.addi %seq, %c1 : index
    // expected-error @+1 {{dynamic tensor.empty size must be tensor.dim of entry argument 0 at axis 1}}
    %empty = tensor.empty(%other) : tensor<?x64xf32>
    %out = tensor.expand_shape %empty [[0, 1], [2]] output_shape [1, %seq, 64]
        : tensor<?x64xf32> into tensor<1x?x64xf32>
    return %out : tensor<1x?x64xf32>
  }
}
