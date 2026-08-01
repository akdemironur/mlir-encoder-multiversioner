// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   --split-input-file -verify-diagnostics %s -o /dev/null

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %c1 = arith.constant 1 : index
    %s = tensor.dim %ids, %c1 : tensor<1x?xi64>
    // expected-error @+1 {{unsupported dynamic tensor result type 'tensor<1x?x2xf32>'}}
    %ambiguous = tensor.empty(%s) : tensor<1x?x2xf32>
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %c1 = arith.constant 1 : index
    %s = tensor.dim %ids, %c1 : tensor<1x?xi64>
    %positions = tensor.empty() : tensor<1x512xi64>
    // expected-error @+1 {{unsupported E5 tensor.extract_slice pattern}}
    %static_slice = tensor.extract_slice %positions[0, 0] [1, 16] [1, 1]
        : tensor<1x512xi64> to tensor<1x16xi64>
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %c1 = arith.constant 1 : index
    %s = tensor.dim %ids, %c1 : tensor<1x?xi64>
    %hidden = tensor.empty(%s) : tensor<1x?x384xf32>
    // expected-error @+1 {{unsupported dynamic tensor operation tensor.cast}}
    %unsupported = tensor.cast %hidden
        : tensor<1x?x384xf32> to tensor<1x?x384xf32>
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %c1 = arith.constant 1 : index
    %s = tensor.dim %ids, %c1 : tensor<1x?xi64>
    %other = arith.addi %s, %c1 : index
    // expected-error @+1 {{dynamic tensor.empty size is not proven equal to the E5 entry sequence dimension}}
    %unproven = tensor.empty(%other) : tensor<1x?x384xf32>
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  func.func private @helper(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32> {
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }

  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %callee = func.constant @helper
        : (tensor<1x?xi64>, tensor<1x?xi64>, tensor<1x?xi64>)
            -> tensor<1x384xf32>
    // expected-error @+1 {{E5-small-v2 does not support indirect calls}}
    %result = func.call_indirect %callee(%ids, %mask, %types)
        : (tensor<1x?xi64>, tensor<1x?xi64>, tensor<1x?xi64>)
            -> tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    // expected-error @+1 {{E5-small-v2 does not support recursion}}
    %result = func.call @sentence_embedding(%ids, %mask, %types)
        : (tensor<1x?xi64>, tensor<1x?xi64>, tensor<1x?xi64>)
            -> tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %c1 = arith.constant 1 : index
    %s = tensor.dim %ids, %c1 : tensor<1x?xi64>
    %hidden = tensor.empty(%s) : tensor<1x?x384xf32>
    // expected-error @+1 {{unsupported E5 tensor.collapse_shape type pair}}
    %flat = tensor.collapse_shape %hidden [[0, 1], [2]]
        : tensor<1x?x384xf32> into tensor<?x384xf32>
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}

// -----

module {
  func.func @sentence_embedding(
      %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
      %types: tensor<1x?xi64>) -> tensor<1x384xf32>
      attributes {shortseq.entry, shortseq.e5_small_v2} {
    %c1 = arith.constant 1 : index
    %s = tensor.dim %ids, %c1 : tensor<1x?xi64>
    %positions = tensor.empty() : tensor<512xi64>
    // expected-error @+1 {{unsupported E5 tensor.extract_slice pattern}}
    %slice = tensor.extract_slice %positions[0] [%s] [1]
        : tensor<512xi64> to tensor<?xi64>
    %result = tensor.empty() : tensor<1x384xf32>
    return %result : tensor<1x384xf32>
  }
}
