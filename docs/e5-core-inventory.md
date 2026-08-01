# E5 core operation and type inventory

This file is generated from the generic-form textual core by
`scripts/inventory_e5_core.py` and then manually reviewed. It records
the pinned model revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`.
Counts are textual occurrences in that deterministic bridge artifact,
not runtime execution counts.

Every `?` axis below is classified as the one entry sequence dimension
`S`; every other axis is static. The pass contains the corresponding
explicit allowlist and rejects any family not shown here.

## Operations

| Operation | Count |
| --- | ---: |
| `arith.addf` | 333 |
| `arith.addi` | 4 |
| `arith.cmpf` | 12 |
| `arith.cmpi` | 11 |
| `arith.constant` | 17 |
| `arith.divf` | 113 |
| `arith.index_cast` | 16 |
| `arith.maximumf` | 12 |
| `arith.mulf` | 148 |
| `arith.select` | 19 |
| `arith.sitofp` | 2 |
| `arith.subf` | 38 |
| `builtin.module` | 1 |
| `cf.assert` | 63 |
| `func.func` | 1 |
| `func.return` | 1 |
| `linalg.batch_matmul` | 96 |
| `linalg.fill` | 10 |
| `linalg.generic` | 613 |
| `linalg.index` | 15 |
| `linalg.transpose` | 48 |
| `linalg.yield` | 767 |
| `math.erf` | 12 |
| `math.exp` | 12 |
| `math.fpowi` | 25 |
| `math.sqrt` | 26 |
| `tensor.collapse_shape` | 63 |
| `tensor.dim` | 3 |
| `tensor.empty` | 28 |
| `tensor.expand_shape` | 77 |
| `tensor.extract` | 3 |
| `tensor.extract_slice` | 1 |
| `util.global` | 198 |
| `util.global.load` | 198 |

## Dynamic tensor families

| Tensor type | Occurrences | `S` axes |
| --- | ---: | --- |
| `tensor<12x32x?xf32>` | 24 | 2 |
| `tensor<12x?x32xf32>` | 87 | 1 |
| `tensor<12x?x?xf32>` | 63 | 1, 2 |
| `tensor<1x12x32x?xf32>` | 37 | 3 |
| `tensor<1x12x?x1xf32>` | 63 | 2 |
| `tensor<1x12x?x32xf32>` | 97 | 2 |
| `tensor<1x12x?x?xf32>` | 229 | 2, 3 |
| `tensor<1x12x?xf32>` | 39 | 2 |
| `tensor<1x12x?xi64>` | 27 | 2 |
| `tensor<1x1x1x?xf32>` | 21 | 3 |
| `tensor<1x1x1x?xi64>` | 2 | 3 |
| `tensor<1x?x12x32xf32>` | 109 | 1 |
| `tensor<1x?x1536xf32>` | 267 | 1 |
| `tensor<1x?x1xf32>` | 455 | 1 |
| `tensor<1x?x384xf32>` | 947 | 1 |
| `tensor<1x?xf32>` | 5 | 1 |
| `tensor<1x?xi1>` | 12 | 1 |
| `tensor<1x?xi64>` | 42 | 1 |
| `tensor<?x384xf32>` | 12 | 0 |
| `tensor<?xi64>` | 6 | 0 |
