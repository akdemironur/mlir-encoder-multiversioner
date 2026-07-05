module {
  func.func @stage_b_preembed_adapter(
      %embedded: tensor<1x?x64xf32>
  ) -> tensor<1x?x64xf32> {
    return %embedded : tensor<1x?x64xf32>
  }

  func.func @run_encoder(
      %embedded: tensor<1x?x64xf32>,
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
  ) -> tensor<1x?x64xf32> {
    %x = func.call @stage_b_preembed_adapter(%embedded)
        : (tensor<1x?x64xf32>) -> tensor<1x?x64xf32>
    %y = func.call @encoder(
        %x, %q_w, %k_w, %v_w, %o_w, %norm_scale, %norm_bias, %ff_w1, %ff_b1,
        %ff_w2, %ff_b2)
        : (tensor<1x?x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>,
           tensor<64x64xf32>, tensor<64x64xf32>, tensor<64xf32>,
           tensor<64xf32>, tensor<64x256xf32>, tensor<256xf32>,
           tensor<256x64xf32>, tensor<64xf32>) -> tensor<1x?x64xf32>
    return %y : tensor<1x?x64xf32>
  }

  func.func @encoder(
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
