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
    %zero = arith.constant 0.0 : f32
    %seq = tensor.dim %x, %c1 : tensor<1x?x64xf32>

    %x_2d = tensor.collapse_shape %x [[0, 1], [2]]
        : tensor<1x?x64xf32> into tensor<?x64xf32>

    %q_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %q_init = linalg.fill ins(%zero : f32) outs(%q_empty : tensor<?x64xf32>)
        -> tensor<?x64xf32>
    %q = linalg.matmul
        ins(%x_2d, %q_w : tensor<?x64xf32>, tensor<64x64xf32>)
        outs(%q_init : tensor<?x64xf32>) -> tensor<?x64xf32>

    %k_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %k_init = linalg.fill ins(%zero : f32) outs(%k_empty : tensor<?x64xf32>)
        -> tensor<?x64xf32>
    %k = linalg.matmul
        ins(%x_2d, %k_w : tensor<?x64xf32>, tensor<64x64xf32>)
        outs(%k_init : tensor<?x64xf32>) -> tensor<?x64xf32>

    %v_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %v_init = linalg.fill ins(%zero : f32) outs(%v_empty : tensor<?x64xf32>)
        -> tensor<?x64xf32>
    %v = linalg.matmul
        ins(%x_2d, %v_w : tensor<?x64xf32>, tensor<64x64xf32>)
        outs(%v_init : tensor<?x64xf32>) -> tensor<?x64xf32>

    %mix_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %mix = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (s, h)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%q, %k, %v : tensor<?x64xf32>, tensor<?x64xf32>, tensor<?x64xf32>)
    outs(%mix_empty : tensor<?x64xf32>) {
    ^bb0(%qv: f32, %kv: f32, %vv: f32, %unused: f32):
      %qk = arith.addf %qv, %kv : f32
      %out = arith.addf %qk, %vv : f32
      linalg.yield %out : f32
    } -> tensor<?x64xf32>

    %attn_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %attn_init = linalg.fill ins(%zero : f32)
                              outs(%attn_empty : tensor<?x64xf32>)
                              -> tensor<?x64xf32>
    %attn = linalg.matmul
        ins(%mix, %o_w : tensor<?x64xf32>, tensor<64x64xf32>)
        outs(%attn_init : tensor<?x64xf32>) -> tensor<?x64xf32>

    %residual_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %residual = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (s, h)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%attn, %x_2d : tensor<?x64xf32>, tensor<?x64xf32>)
    outs(%residual_empty : tensor<?x64xf32>) {
    ^bb0(%av: f32, %xv: f32, %unused: f32):
      %out = arith.addf %av, %xv : f32
      linalg.yield %out : f32
    } -> tensor<?x64xf32>

    %norm_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %norm = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (h)>,
        affine_map<(s, h) -> (h)>,
        affine_map<(s, h) -> (s, h)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%residual, %norm_scale, %norm_bias
        : tensor<?x64xf32>, tensor<64xf32>, tensor<64xf32>)
    outs(%norm_empty : tensor<?x64xf32>) {
    ^bb0(%value: f32, %scale: f32, %bias: f32, %unused: f32):
      %scaled = arith.mulf %value, %scale : f32
      %out = arith.addf %scaled, %bias : f32
      linalg.yield %out : f32
    } -> tensor<?x64xf32>

    %ff1_empty = tensor.empty(%seq) : tensor<?x256xf32>
    %ff1_init = linalg.fill ins(%zero : f32) outs(%ff1_empty : tensor<?x256xf32>)
        -> tensor<?x256xf32>
    %ff1_mm = linalg.matmul
        ins(%norm, %ff_w1 : tensor<?x64xf32>, tensor<64x256xf32>)
        outs(%ff1_init : tensor<?x256xf32>) -> tensor<?x256xf32>

    %ff1_act_empty = tensor.empty(%seq) : tensor<?x256xf32>
    %ff1 = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (h)>,
        affine_map<(s, h) -> (s, h)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%ff1_mm, %ff_b1 : tensor<?x256xf32>, tensor<256xf32>)
    outs(%ff1_act_empty : tensor<?x256xf32>) {
    ^bb0(%value: f32, %bias: f32, %unused: f32):
      %biased = arith.addf %value, %bias : f32
      %out = arith.mulf %biased, %biased : f32
      linalg.yield %out : f32
    } -> tensor<?x256xf32>

    %ff2_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %ff2_init = linalg.fill ins(%zero : f32) outs(%ff2_empty : tensor<?x64xf32>)
        -> tensor<?x64xf32>
    %ff2_mm = linalg.matmul
        ins(%ff1, %ff_w2 : tensor<?x256xf32>, tensor<256x64xf32>)
        outs(%ff2_init : tensor<?x64xf32>) -> tensor<?x64xf32>

    %out_empty = tensor.empty(%seq) : tensor<?x64xf32>
    %out_2d = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (h)>,
        affine_map<(s, h) -> (s, h)>,
        affine_map<(s, h) -> (s, h)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%ff2_mm, %ff_b2, %residual
        : tensor<?x64xf32>, tensor<64xf32>, tensor<?x64xf32>)
    outs(%out_empty : tensor<?x64xf32>) {
    ^bb0(%value: f32, %bias: f32, %skip: f32, %unused: f32):
      %biased = arith.addf %value, %bias : f32
      %out = arith.addf %biased, %skip : f32
      linalg.yield %out : f32
    } -> tensor<?x64xf32>

    %out = tensor.expand_shape %out_2d [[0, 1], [2]]
        output_shape [1, %seq, 64]
        : tensor<?x64xf32> into tensor<1x?x64xf32>

    return %out : tensor<1x?x64xf32>
  }
}
