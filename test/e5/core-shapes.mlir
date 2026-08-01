// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=16})' \
// RUN:   %s | %FileCheck %s

#token = affine_map<(b, s, h) -> (b, s)>
#hidden = affine_map<(b, s, h) -> (b, s, h)>
#pooled = affine_map<(b, s, h) -> (b, h)>

// A small structural fixture for the three model regions that carry S:
// token embedding, both attention-score axes, and masked pooling.
module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    %s = tensor.dim %ids, %c1 : tensor<1x?xi64>

    %embedding_init = tensor.empty(%s) : tensor<1x?x384xf32>
    %embedding = linalg.generic {
      indexing_maps = [#token, #token, #hidden],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%ids, %types : tensor<1x?xi64>, tensor<1x?xi64>)
      outs(%embedding_init : tensor<1x?x384xf32>) {
    ^bb0(%id: i64, %type: i64, %unused: f32):
      %id_f = arith.sitofp %id : i64 to f32
      %type_f = arith.sitofp %type : i64 to f32
      %sum = arith.addf %id_f, %type_f : f32
      linalg.yield %sum : f32
    } -> tensor<1x?x384xf32>

    %heads = tensor.expand_shape %embedding [[0], [1], [2, 3]]
        output_shape [1, %s, 12, 32]
        : tensor<1x?x384xf32> into tensor<1x?x12x32xf32>
    %query_init = tensor.empty(%s) : tensor<1x12x?x32xf32>
    %query = linalg.transpose
        ins(%heads : tensor<1x?x12x32xf32>)
        outs(%query_init : tensor<1x12x?x32xf32>)
        permutation = [0, 2, 1, 3]
    %key_init = tensor.empty(%s) : tensor<1x12x32x?xf32>
    %key = linalg.transpose
        ins(%query : tensor<1x12x?x32xf32>)
        outs(%key_init : tensor<1x12x32x?xf32>)
        permutation = [0, 1, 3, 2]
    %query_3d = tensor.collapse_shape %query [[0, 1], [2], [3]]
        : tensor<1x12x?x32xf32> into tensor<12x?x32xf32>
    %key_3d = tensor.collapse_shape %key [[0, 1], [2], [3]]
        : tensor<1x12x32x?xf32> into tensor<12x32x?xf32>
    %scores_init = tensor.empty(%s, %s) : tensor<12x?x?xf32>
    %scores = linalg.batch_matmul
        ins(%query_3d, %key_3d : tensor<12x?x32xf32>, tensor<12x32x?xf32>)
        outs(%scores_init : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    %scores_4d = tensor.expand_shape %scores [[0, 1], [2], [3]]
        output_shape [1, 12, %s, %s]
        : tensor<12x?x?xf32> into tensor<1x12x?x?xf32>
    %scores_3d = tensor.collapse_shape %scores_4d [[0, 1], [2], [3]]
        : tensor<1x12x?x?xf32> into tensor<12x?x?xf32>
    %context_init = tensor.empty(%s) : tensor<12x?x32xf32>
    %context = linalg.batch_matmul
        ins(%scores_3d, %query_3d
            : tensor<12x?x?xf32>, tensor<12x?x32xf32>)
        outs(%context_init : tensor<12x?x32xf32>) -> tensor<12x?x32xf32>
    %context_4d = tensor.expand_shape %context [[0, 1], [2], [3]]
        output_shape [1, 12, %s, 32]
        : tensor<12x?x32xf32> into tensor<1x12x?x32xf32>
    %sequence_init = tensor.empty(%s) : tensor<1x?x12x32xf32>
    %sequence = linalg.transpose
        ins(%context_4d : tensor<1x12x?x32xf32>)
        outs(%sequence_init : tensor<1x?x12x32xf32>)
        permutation = [0, 2, 1, 3]
    %hidden_state = tensor.collapse_shape %sequence [[0], [1], [2, 3]]
        : tensor<1x?x12x32xf32> into tensor<1x?x384xf32>

    %pooled_init = tensor.empty() : tensor<1x384xf32>
    %pooled_zero = linalg.fill ins(%c0 : f32)
        outs(%pooled_init : tensor<1x384xf32>) -> tensor<1x384xf32>
    %pooled_result = linalg.generic {
      indexing_maps = [#hidden, #token, #pooled],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%hidden_state, %mask : tensor<1x?x384xf32>, tensor<1x?xi64>)
      outs(%pooled_zero : tensor<1x384xf32>) {
    ^bb0(%value: f32, %mask_value: i64, %sum: f32):
      %mask_f = arith.sitofp %mask_value : i64 to f32
      %masked = arith.mulf %value, %mask_f : f32
      %next = arith.addf %sum, %masked : f32
      linalg.yield %next : f32
    } -> tensor<1x384xf32>
    return %pooled_result : tensor<1x384xf32>
  }
}

// CHECK-LABEL: func.func @sentence_embedding_generic(
// CHECK: tensor<12x?x?xf32>
// CHECK-LABEL: func.func private @sentence_embedding_s16(
// CHECK-NOT: ?
// CHECK: tensor<12x16x16xf32>
// CHECK: tensor<1x16x384xf32>
// CHECK: return {{.*}} : tensor<1x384xf32>
// CHECK-LABEL: func.func @sentence_embedding(
