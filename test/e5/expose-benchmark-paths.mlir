// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=16})' \
// RUN:   %s | %FileCheck %s --check-prefix=DEFAULT
// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=16 expose-static-variants=true})' \
// RUN:   %s | %FileCheck %s --check-prefix=EXPOSED

// Benchmarking may expose the pass-produced static clone as a second public
// entry. The default output remains unchanged.
module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// DEFAULT-LABEL: func.func private @sentence_embedding_s16(
// DEFAULT-LABEL: func.func @sentence_embedding(

// EXPOSED-LABEL: func.func @sentence_embedding_s16(
// EXPOSED-SAME: tensor<1x16xi64>
// EXPOSED-SAME: tensor<1x16xi64>
// EXPOSED-SAME: tensor<1x16xi64>
// EXPOSED-LABEL: func.func @sentence_embedding(
// EXPOSED: func.call @sentence_embedding_s16
