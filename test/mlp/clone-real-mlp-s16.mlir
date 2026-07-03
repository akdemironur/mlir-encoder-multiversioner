// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize)' \
// RUN:   %S/../../examples/mlp/dynamic_mlp.mlir | %FileCheck %s

// CHECK-LABEL: func.func @mlp_generic(
// CHECK-SAME: tensor<1x?x384xf32>
// CHECK: tensor.empty(%{{.*}}) : tensor<?x1536xf32>

// CHECK-LABEL: func.func private @mlp_s16(
// CHECK-SAME: %[[X:.*]]: tensor<1x16x384xf32>
// CHECK-SAME: tensor<384x1536xf32>
// CHECK-SAME: tensor<1536xf32>
// CHECK-SAME: tensor<1536x384xf32>
// CHECK-SAME: tensor<384xf32>
// CHECK-SAME: ) -> tensor<1x16x384xf32>
// CHECK-NOT: tensor<?
// CHECK: tensor.collapse_shape %[[X]]
// CHECK-SAME: : tensor<1x16x384xf32> into tensor<16x384xf32>
// CHECK: tensor.empty() : tensor<16x1536xf32>
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x384xf32>, tensor<384x1536xf32>
// CHECK-SAME: tensor<16x1536xf32>
// CHECK: tensor.empty() : tensor<16x384xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME: output_shape [1, 16, 384]
// CHECK-SAME: : tensor<16x384xf32> into tensor<1x16x384xf32>
// CHECK: return %{{.*}} : tensor<1x16x384xf32>
