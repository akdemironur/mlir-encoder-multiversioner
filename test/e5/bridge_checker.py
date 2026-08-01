#!/usr/bin/env python3
"""Exercise the bridge checker's dense-payload rejection."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_e5_bridge import check_no_dense_payloads  # noqa: E402


def expect_rejection(core: str) -> None:
    try:
        check_no_dense_payloads(core)
    except AssertionError:
        return
    raise AssertionError("expected embedded dense payload rejection")


def main() -> int:
    check_no_dense_payloads("%cst = arith.constant dense<1.0> : tensor<f32>")
    expect_rejection(
        "%weights = arith.constant dense<0.0> : tensor<30522x384xf32>"
    )
    expect_rejection(
        '%weights = "arith.constant"() {value = dense_resource<weights>} '
        ": () -> tensor<30522x384xf32>"
    )
    print("PASS E5 bridge checker rejects embedded dense payloads")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
