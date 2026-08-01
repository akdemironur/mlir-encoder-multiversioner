#!/usr/bin/env python3
"""Check the symbolic MLIR references and canonical bytes in a Stage B bundle."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import iree.runtime as ireert
except ImportError as exc:
    raise SystemExit(
        "IREE runtime is required. Run with: "
        "uv run --frozen --group bench python scripts/check_stage_b_bundle.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAM_BYTES = {
    "q_w": 64 * 64 * 4,
    "k_w": 64 * 64 * 4,
    "v_w": 64 * 64 * 4,
    "o_w": 64 * 64 * 4,
    "norm_scale": 64 * 4,
    "norm_bias": 64 * 4,
    "ff_w1": 64 * 256 * 4,
    "ff_b1": 256 * 4,
    "ff_w2": 256 * 64 * 4,
    "ff_b2": 64 * 4,
}
REFERENCE_RE = re.compile(
    r'#flow\.parameter\.named<"(?P<scope>[^"]+)"::"(?P<key>[^"]+)">'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlir",
        default=REPO_ROOT / "results/stage_b/encoder_block_bundle.mlir",
        type=Path,
    )
    parser.add_argument(
        "--irpa",
        default=REPO_ROOT / "results/stage_b/encoder_block_weights.irpa",
        type=Path,
    )
    parser.add_argument("--scope", default="stage_b")
    return parser.parse_args()


def check_mlir(path: Path, scope: str) -> None:
    ir = path.read_text()
    if "dense<" in ir:
        raise AssertionError("bundle MLIR contains embedded dense payloads")

    references = REFERENCE_RE.findall(ir)
    expected = [(scope, name) for name in PARAM_BYTES]
    if references != expected:
        raise AssertionError(
            f"parameter references {references}, expected canonical order {expected}"
        )
    if ir.count('"util.global.load"') != len(PARAM_BYTES):
        raise AssertionError("expected exactly one global load per parameter")
    if "func.func @run_encoder_block(" not in ir:
        raise AssertionError("missing public @run_encoder_block adapter")


def check_irpa(path: Path) -> int:
    index = ireert.ParameterIndex()
    index.load(str(path))
    entries = {key: entry for key, entry in index.items()}
    if set(entries) != set(PARAM_BYTES):
        raise AssertionError(
            f"archive keys {sorted(entries)}, expected {sorted(PARAM_BYTES)}"
        )
    for name, expected_bytes in PARAM_BYTES.items():
        if entries[name].length != expected_bytes:
            raise AssertionError(
                f"{name} bytes {entries[name].length}, expected {expected_bytes}"
            )
    return sum(entry.length for entry in entries.values())


def main() -> int:
    args = parse_args()
    check_mlir(args.mlir, args.scope)
    canonical_bytes = check_irpa(args.irpa)
    print(f"PASS Stage B bundle: {args.mlir} + {args.irpa}")
    print(f"canonical_parameter_bytes={canonical_bytes}")
    print("duplicated_parameter_bytes=0")
    print("prepacked_weight_bytes=not_measured")
    print("executable_code_bytes=not_built")
    print("variant_metadata_bytes=not_built")
    print("active_scratch_bytes=not_measured")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
