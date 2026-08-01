// RUN: %not %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=513})' \
// RUN:   %s 2>&1 | %FileCheck %s

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// CHECK: error: E5-small-v2 specialization length 513 exceeds the position-embedding limit 512
