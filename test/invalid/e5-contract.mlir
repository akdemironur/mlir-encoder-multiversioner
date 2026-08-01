// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   --split-input-file -verify-diagnostics %s -o /dev/null

module {
  // expected-error @+1 {{expected E5-small-v2 entry to have 3 arguments}}
  func.func @sentence_embedding(%ids: tensor<1x?xi64>,
      %mask: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected input_ids argument 0 to have rank 2}}
  func.func @sentence_embedding(%ids: tensor<1x?x1xi64>,
      %mask: tensor<1x?xi64>, %types: tensor<1x?xi64>)
      -> tensor<1x384xf32> attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected attention_mask argument 1 element type i64}}
  func.func @sentence_embedding(%ids: tensor<1x?xi64>,
      %mask: tensor<1x?xi32>, %types: tensor<1x?xi64>)
      -> tensor<1x384xf32> attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected token_type_ids argument 2 batch dimension 1, got 2}}
  func.func @sentence_embedding(%ids: tensor<1x?xi64>,
      %mask: tensor<1x?xi64>, %types: tensor<2x?xi64>)
      -> tensor<1x384xf32> attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected result 0 to have type 'tensor<1x384xf32>'}}
  func.func @sentence_embedding(%ids: tensor<1x?xi64>,
      %mask: tensor<1x?xi64>, %types: tensor<1x?xi64>)
      -> tensor<1x385xf32> attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x385xf32>
    return %result : tensor<1x385xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected input_ids argument 0 batch dimension 1, got dynamic}}
  func.func @sentence_embedding(%ids: tensor<?x16xi64>,
      %mask: tensor<1x?xi64>, %types: tensor<1x?xi64>)
      -> tensor<1x384xf32> attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected token_type_ids argument 2 dynamic sequence dimension at axis 1}}
  func.func @sentence_embedding(%ids: tensor<1x?xi64>,
      %mask: tensor<1x?xi64>, %types: tensor<1x16xi64>)
      -> tensor<1x384xf32> attributes {shortseq.entry, shortseq.e5_small_v2} {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}
