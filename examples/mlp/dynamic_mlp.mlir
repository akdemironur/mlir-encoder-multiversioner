module {
  func.func @mlp(
      %x: tensor<1x?x384xf32>,
      %w1: tensor<384x1536xf32>,
      %b1: tensor<1536xf32>,
      %w2: tensor<1536x384xf32>,
      %b2: tensor<384xf32>
  ) -> tensor<1x?x384xf32> attributes {shortseq.entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %zero = arith.constant 0.0 : f32
    %half = arith.constant 0.5 : f32
    %one = arith.constant 1.0 : f32
    %inv_sqrt_2 = arith.constant 0.7071067811865476 : f32

    %seq = tensor.dim %x, %c1 : tensor<1x?x384xf32>

    // [1, S, 384] -> [S, 384].
    %x_2d = tensor.collapse_shape %x [[0, 1], [2]]
        : tensor<1x?x384xf32> into tensor<?x384xf32>

    // hidden_pre = x * w1
    // [S, 384] x [384, 1536] -> [S, 1536]

    %hidden_empty = tensor.empty(%seq) : tensor<?x1536xf32>
    %hidden_init = linalg.fill ins(%zero : f32)
                               outs(%hidden_empty : tensor<?x1536xf32>)
                               -> tensor<?x1536xf32>

    %hidden_mm = linalg.matmul
        ins(%x_2d, %w1 : tensor<?x384xf32>, tensor<384x1536xf32>)
        outs(%hidden_init : tensor<?x1536xf32>)
        -> tensor<?x1536xf32>

    // hidden = GELU(hidden_pre + b1)
    // Bias is broadcast across the dynamic sequence dimension.

    %hidden_act_empty = tensor.empty(%seq) : tensor<?x1536xf32>

    %hidden = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>, // hidden_mm[s, h]
        affine_map<(s, h) -> (h)>,    // b1[h]
        affine_map<(s, h) -> (s, h)>  // output[s, h]
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%hidden_mm, %b1 : tensor<?x1536xf32>, tensor<1536xf32>)
    outs(%hidden_act_empty : tensor<?x1536xf32>) {
    ^bb0(%value: f32, %bias: f32, %unused: f32):
      %z = arith.addf %value, %bias : f32

      // Exact GELU:
      // 0.5 * z * (1 + erf(z / sqrt(2)))
      %scaled = arith.mulf %z, %inv_sqrt_2 : f32
      %erf = math.erf %scaled : f32
      %cdf_term = arith.addf %one, %erf : f32
      %product = arith.mulf %z, %cdf_term : f32
      %gelu = arith.mulf %half, %product : f32

      linalg.yield %gelu : f32
    } -> tensor<?x1536xf32>

    // y_2d = hidden * w2
    // [S, 1536] x [1536, 384] -> [S, 384]

    %y_empty = tensor.empty(%seq) : tensor<?x384xf32>
    %y_init = linalg.fill ins(%zero : f32)
                          outs(%y_empty : tensor<?x384xf32>)
                          -> tensor<?x384xf32>

    %y_mm = linalg.matmul
        ins(%hidden, %w2 : tensor<?x1536xf32>, tensor<1536x384xf32>)
        outs(%y_init : tensor<?x384xf32>)
        -> tensor<?x384xf32>

    // Final bias broadcast: [S, 384] + [384].
    %y_biased_empty = tensor.empty(%seq) : tensor<?x384xf32>

    %y_2d = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>, // y_mm[s, h]
        affine_map<(s, h) -> (h)>,    // b2[h]
        affine_map<(s, h) -> (s, h)>  // output[s, h]
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%y_mm, %b2 : tensor<?x384xf32>, tensor<384xf32>)
    outs(%y_biased_empty : tensor<?x384xf32>) {
    ^bb0(%value: f32, %bias: f32, %unused: f32):
      %result = arith.addf %value, %bias : f32
      linalg.yield %result : f32
    } -> tensor<?x384xf32>

    // Residual: y_2d + x_2d

    %residual_empty = tensor.empty(%seq) : tensor<?x384xf32>
    %residual_2d = linalg.generic {
      indexing_maps = [
        affine_map<(s, h) -> (s, h)>, // y_2d[s, h]
        affine_map<(s, h) -> (s, h)>, // x_2d[s, h]
        affine_map<(s, h) -> (s, h)>  // output[s, h]
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%y_2d, %x_2d : tensor<?x384xf32>, tensor<?x384xf32>)
    outs(%residual_empty : tensor<?x384xf32>) {
    ^bb0(%y_val: f32, %x_val: f32, %unused: f32):
      %result = arith.addf %y_val, %x_val : f32
      linalg.yield %result : f32
    } -> tensor<?x384xf32>

    // [S, 384] -> [1, S, 384].
    %y = tensor.expand_shape %residual_2d [[0, 1], [2]]
        output_shape [1, %seq, 384]
        : tensor<?x384xf32> into tensor<1x?x384xf32>

    return %y : tensor<1x?x384xf32>
  }
}
