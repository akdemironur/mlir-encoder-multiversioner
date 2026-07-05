# Stage B Contract

Stage B targets an EmbeddingGemma-like encoder block, but v1 does not import a
real model. The pass sees pre-embedded f32 activations and shared parameter
operands only.

## Entry

The specialized entry is marked with:

```mlir
attributes {shortseq.entry, shortseq.stage_b}
```

It must have one dynamic sequence axis:

```mlir
tensor<1x?x64xf32>
```

The result is token states with the same dynamic shape:

```mlir
tensor<1x?x64xf32>
```

Pooling is out of scope.

## Parameters

The entry has exactly these parameter operands after the input:

```text
q_w        tensor<64x64xf32>
k_w        tensor<64x64xf32>
v_w        tensor<64x64xf32>
o_w        tensor<64x64xf32>
norm_scale tensor<64xf32>
norm_bias  tensor<64xf32>
ff_w1      tensor<64x256xf32>
ff_b1      tensor<256xf32>
ff_w2      tensor<256x64xf32>
ff_b2      tensor<64xf32>
```

Parameters are forwarded to generic and static variants. The pass must not
clone dense learned-weight payloads.

## Dynamic Shapes

The only dynamic tensor shapes currently accepted in the body are:

```mlir
tensor<1x?x64xf32>
tensor<?x64xf32>
tensor<?x256xf32>
```

Dynamic allocation must use the selected sequence value:

```mlir
%c1 = arith.constant 1 : index
%seq = tensor.dim %x, %c1 : tensor<1x?x64xf32>
%empty = tensor.empty(%seq) : tensor<?x64xf32>
```

The pass rejects dynamic shapes derived from any other producer.

## Adapter Boundary

Token IDs, embedding lookup, masks, padding policy, and model artifact loading
stay outside this pass in v1. An adapter may produce the pre-embedded
`tensor<1x?x64xf32>` activation, but that adapter is unmarked and is not
specialized by `shortseq-specialize`.

Adding token IDs, masks, pooling, extra parameters, or different hidden sizes is
a contract change. It needs a design note, positive and negative IR tests,
README updates, and benchmark/accounting updates.
