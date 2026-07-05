// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8,16})' \
// RUN:   %S/../../examples/stage_b/tiny_gemma_encoder.mlir | %FileCheck --check-prefix=TINY %s
// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8})' \
// RUN:   %S/../../examples/stage_b/preembed_adapter_boundary.mlir | %FileCheck --check-prefix=ADAPTER %s
// RUN: %mlir_opt \
// RUN:   --load-pass-plugin=%shortseq_plugin \
// RUN:   --pass-pipeline='builtin.module(shortseq-specialize{lengths=4,8})' \
// RUN:   %S/../../examples/stage_b/contract_encoder_block.mlir | %FileCheck --check-prefix=BLOCK %s

// TINY: module attributes {shortseq.ran} {
// TINY-NOT: dense<
// TINY-LABEL: func.func @encoder_generic(
// TINY-SAME: tensor<1x?x64xf32>
// TINY-NOT: shortseq.stage_b
// TINY-LABEL: func.func private @encoder_s4(
// TINY-SAME: tensor<1x4x64xf32>
// TINY-LABEL: func.func private @encoder_s8(
// TINY-SAME: tensor<1x8x64xf32>
// TINY-LABEL: func.func private @encoder_s16(
// TINY-SAME: tensor<1x16x64xf32>
// TINY-LABEL: func.func @encoder(
// TINY-SAME: tensor<1x?x64xf32>
// TINY-NOT: shortseq.stage_b
// TINY: tensor.dim
// TINY: func.call @encoder_s4
// TINY: func.call @encoder_s8
// TINY: func.call @encoder_s16
// TINY: func.call @encoder_generic

// ADAPTER-LABEL: func.func @stage_b_preembed_adapter(
// ADAPTER-LABEL: func.func @run_encoder(
// ADAPTER: call @stage_b_preembed_adapter
// ADAPTER: call @encoder(
// ADAPTER-LABEL: func.func @encoder_generic(
// ADAPTER-LABEL: func.func private @encoder_s4(
// ADAPTER-SAME: tensor<1x4x64xf32>
// ADAPTER-LABEL: func.func private @encoder_s8(
// ADAPTER-SAME: tensor<1x8x64xf32>
// ADAPTER-LABEL: func.func @encoder(
// ADAPTER-SAME: tensor<1x?x64xf32>

// BLOCK: module attributes {shortseq.ran} {
// BLOCK-NOT: dense<
// BLOCK-LABEL: func.func @encoder_block_generic(
// BLOCK-SAME: tensor<1x?x64xf32>
// BLOCK-LABEL: func.func private @encoder_block_s4(
// BLOCK-SAME: tensor<1x4x64xf32>
// BLOCK-LABEL: func.func private @encoder_block_s8(
// BLOCK-SAME: tensor<1x8x64xf32>
// BLOCK-LABEL: func.func @encoder_block(
// BLOCK-SAME: tensor<1x?x64xf32>
// BLOCK: tensor.dim
// BLOCK: func.call @encoder_block_s4
// BLOCK: func.call @encoder_block_s8
// BLOCK: func.call @encoder_block_generic
