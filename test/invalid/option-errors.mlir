// RUN: %not %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=0})' \
// RUN:   %s 2>&1 | %FileCheck --check-prefix=BAD-LENGTH %s
// RUN: %not %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=16,16})' \
// RUN:   %s 2>&1 | %FileCheck --check-prefix=DUP-LENGTH %s

module {
  func.func @mlp(%x: tensor<1x?x384xf32>, %w1: tensor<384x1536xf32>,
                 %b1: tensor<1536xf32>, %w2: tensor<1536x384xf32>,
                 %b2: tensor<384xf32>) -> tensor<1x?x384xf32>
      attributes {shortseq.entry} {
    return %x : tensor<1x?x384xf32>
  }
}

// BAD-LENGTH: error: shortseq-specialize --lengths expects positive integers, got `0`
// DUP-LENGTH: error: shortseq-specialize --lengths contains duplicate length 16
