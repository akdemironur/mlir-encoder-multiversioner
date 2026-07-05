// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   --split-input-file -verify-diagnostics %s -o /dev/null

// expected-error @+1 {{shortseq-specialize requires exactly one function with shortseq.entry, found 0}}
module {
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x384xf32> {
    return %x : tensor<1x?x384xf32>
  }
}

// -----

// expected-error @+1 {{shortseq-specialize requires exactly one function with shortseq.entry, found 2}}
module {
  func.func @mlp_a(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                   %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                   %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    return %x : tensor<1x?x384xf32>
  }
  func.func @mlp_b(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                   %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                   %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    return %x : tensor<1x?x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{shortseq-specialize requires exactly one dynamic tensor dimension on the entry input}}
  func.func @mlp(%x: tensor<1x?x?xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x?xf32>
      attributes {shortseq.entry} {
    return %x : tensor<1x?x?xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected static batch dimension 1, got 4}}
  func.func @mlp(%x: tensor<4x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<4x?x384xf32>
      attributes {shortseq.entry} {
    return %x : tensor<4x?x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected w1 argument 1 to have type 'tensor<384x1536xf32>'}}
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1535xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    return %x : tensor<1x?x384xf32>
  }
}

// -----

module {
  // expected-error @+1 {{expected result 0 to have type 'tensor<1x?x384xf32>'}}
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x385xf32>
      attributes {shortseq.entry} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x384xf32>
    %empty = tensor.empty(%seq) : tensor<1x?x385xf32>
    return %empty : tensor<1x?x385xf32>
  }
}

// -----

module {
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x384xf32>
    // expected-error @+1 {{unsupported dynamic tensor.empty result type 'tensor<?x512xf32>'}}
    %extra = tensor.empty(%seq) : tensor<?x512xf32>
    return %x : tensor<1x?x384xf32>
  }
}

// -----

module {
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x384xf32>
    %other = arith.addi %seq, %c1 : index
    // expected-error @+1 {{dynamic tensor.empty size must be tensor.dim of entry argument 0 at axis 1}}
    %empty = tensor.empty(%other) : tensor<?x384xf32>
    %y = tensor.expand_shape %empty [[0, 1], [2]] output_shape [1, %seq, 384]
        : tensor<?x384xf32> into tensor<1x?x384xf32>
    return %y : tensor<1x?x384xf32>
  }
}

// -----

module {
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    %c1 = arith.constant 1 : index
    %seq = tensor.dim %x, %c1 : tensor<1x?x384xf32>
    // expected-error @+1 {{unsupported dynamic tensor operation tensor.extract_slice}}
    %slice = tensor.extract_slice %x[0, 0, 0] [1, %seq, 384] [1, 1, 1]
        : tensor<1x?x384xf32> to tensor<1x?x384xf32>
    return %slice : tensor<1x?x384xf32>
  }
}
