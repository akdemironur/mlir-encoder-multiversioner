// The generic spelling lets upstream mlir-opt preserve IREE parameter ops while
// shortseq-specialize runs. IREE recognizes them when compiling the result.
module {
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"q_w"> : tensor<64x64xf32>, sym_name = "stage_b_q_w", sym_visibility = "private", type = tensor<64x64xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"k_w"> : tensor<64x64xf32>, sym_name = "stage_b_k_w", sym_visibility = "private", type = tensor<64x64xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"v_w"> : tensor<64x64xf32>, sym_name = "stage_b_v_w", sym_visibility = "private", type = tensor<64x64xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"o_w"> : tensor<64x64xf32>, sym_name = "stage_b_o_w", sym_visibility = "private", type = tensor<64x64xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"norm_scale"> : tensor<64xf32>, sym_name = "stage_b_norm_scale", sym_visibility = "private", type = tensor<64xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"norm_bias"> : tensor<64xf32>, sym_name = "stage_b_norm_bias", sym_visibility = "private", type = tensor<64xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"ff_w1"> : tensor<64x256xf32>, sym_name = "stage_b_ff_w1", sym_visibility = "private", type = tensor<64x256xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"ff_b1"> : tensor<256xf32>, sym_name = "stage_b_ff_b1", sym_visibility = "private", type = tensor<256xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"ff_w2"> : tensor<256x64xf32>, sym_name = "stage_b_ff_w2", sym_visibility = "private", type = tensor<256x64xf32>}> : () -> ()
  "util.global"() <{initial_value = #flow.parameter.named<"stage_b"::"ff_b2"> : tensor<64xf32>, sym_name = "stage_b_ff_b2", sym_visibility = "private", type = tensor<64xf32>}> : () -> ()

  func.func @run_encoder_block(
      %x: tensor<1x?x64xf32>
  ) -> tensor<1x?x64xf32> {
    %q_w = "util.global.load"() <{global = @stage_b_q_w}> : () -> tensor<64x64xf32>
    %k_w = "util.global.load"() <{global = @stage_b_k_w}> : () -> tensor<64x64xf32>
    %v_w = "util.global.load"() <{global = @stage_b_v_w}> : () -> tensor<64x64xf32>
    %o_w = "util.global.load"() <{global = @stage_b_o_w}> : () -> tensor<64x64xf32>
    %norm_scale = "util.global.load"() <{global = @stage_b_norm_scale}> : () -> tensor<64xf32>
    %norm_bias = "util.global.load"() <{global = @stage_b_norm_bias}> : () -> tensor<64xf32>
    %ff_w1 = "util.global.load"() <{global = @stage_b_ff_w1}> : () -> tensor<64x256xf32>
    %ff_b1 = "util.global.load"() <{global = @stage_b_ff_b1}> : () -> tensor<256xf32>
    %ff_w2 = "util.global.load"() <{global = @stage_b_ff_w2}> : () -> tensor<256x64xf32>
    %ff_b2 = "util.global.load"() <{global = @stage_b_ff_b2}> : () -> tensor<64xf32>
    %y = func.call @encoder_block(
        %x, %q_w, %k_w, %v_w, %o_w, %norm_scale, %norm_bias, %ff_w1, %ff_b1,
        %ff_w2, %ff_b2)
        : (tensor<1x?x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>,
           tensor<64x64xf32>, tensor<64x64xf32>, tensor<64xf32>,
           tensor<64xf32>, tensor<64x256xf32>, tensor<256xf32>,
           tensor<256x64xf32>, tensor<64xf32>) -> tensor<1x?x64xf32>
    return %y : tensor<1x?x64xf32>
  }

  func.func @encoder_block(
      %x: tensor<1x?x64xf32>,
      %q_w: tensor<64x64xf32>,
      %k_w: tensor<64x64xf32>,
      %v_w: tensor<64x64xf32>,
      %o_w: tensor<64x64xf32>,
      %norm_scale: tensor<64xf32>,
      %norm_bias: tensor<64xf32>,
      %ff_w1: tensor<64x256xf32>,
      %ff_b1: tensor<256xf32>,
      %ff_w2: tensor<256x64xf32>,
      %ff_b2: tensor<64xf32>
  ) -> tensor<1x?x64xf32> attributes {shortseq.entry, shortseq.stage_b} {
    return %x : tensor<1x?x64xf32>
  }
}
