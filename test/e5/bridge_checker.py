#!/usr/bin/env python3
"""Exercise the bridge checker's dense-payload rejection."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_e5_bridge import check_dispatch_wrapper, check_no_dense_payloads  # noqa: E402


def expect_rejection(core: str) -> None:
    try:
        check_no_dense_payloads(core)
    except AssertionError:
        return
    raise AssertionError("expected embedded dense payload rejection")


def check_always_generic_wrapper_rejected() -> None:
    core = """
func.func @sentence_embedding(
    %ids: tensor<1x?xi64>, %mask: tensor<1x?xi64>,
    %types: tensor<1x?xi64>) -> tensor<1x384xf32> {
  %c1 = arith.constant 1 : index
  %ids_dim = tensor.dim %ids, %c1 : tensor<1x?xi64>
  %mask_dim = tensor.dim %mask, %c1 : tensor<1x?xi64>
  %types_dim = tensor.dim %types, %c1 : tensor<1x?xi64>
  %c16 = arith.constant 16 : index
  %ids_match = arith.cmpi eq, %ids_dim, %c16 : index
  %mask_match = arith.cmpi eq, %mask_dim, %c16 : index
  %both_match = arith.andi %ids_match, %mask_match : i1
  %types_match = arith.cmpi eq, %types_dim, %c16 : index
  %all_match = arith.andi %both_match, %types_match : i1
  %result = scf.if %all_match -> (tensor<1x384xf32>) {
    %wrong = func.call @sentence_embedding_generic(%ids, %mask, %types)
        : (tensor<1x?xi64>, tensor<1x?xi64>, tensor<1x?xi64>)
          -> tensor<1x384xf32>
    scf.yield %wrong : tensor<1x384xf32>
  } else {
    %fallback = func.call @sentence_embedding_generic(%ids, %mask, %types)
        : (tensor<1x?xi64>, tensor<1x?xi64>, tensor<1x?xi64>)
          -> tensor<1x384xf32>
    scf.yield %fallback : tensor<1x384xf32>
  }
  return %result : tensor<1x384xf32>
}
"""
    try:
        check_dispatch_wrapper(core, [16])
    except AssertionError as exc:
        if "E5 wrapper calls" not in str(exc):
            raise AssertionError(f"unexpected routing diagnostic: {exc}") from exc
        return
    raise AssertionError("artifact checker accepted an always-generic wrapper")


def main() -> int:
    check_no_dense_payloads("%cst = arith.constant dense<1.0> : tensor<f32>")
    expect_rejection(
        "%weights = arith.constant dense<0.0> : tensor<30522x384xf32>"
    )
    expect_rejection(
        '%weights = "arith.constant"() {value = dense_resource<weights>} '
        ": () -> tensor<30522x384xf32>"
    )
    check_always_generic_wrapper_rejected()
    print("PASS E5 bridge checker rejects dense payloads and misrouting")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
