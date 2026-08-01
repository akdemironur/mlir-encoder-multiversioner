#!/usr/bin/env python3
"""Inventory operations and dynamic tensor families in the pinned E5 core."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

from e5_common import DEFAULT_ARTIFACT_DIR, REPO_ROOT, load_manifest


OP_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_.]*)"\(')
TENSOR_RE = re.compile(r"tensor<([^>]+)>")
DEFAULT_INVENTORY = REPO_ROOT / "docs/e5-core-inventory.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "sentence_embedding.generic.mlir",
    )
    parser.add_argument("--check", type=Path)
    return parser.parse_args()


def dynamic_axes(type_body: str) -> list[int]:
    parts = type_body.split("x")
    return [index for index, part in enumerate(parts[:-1]) if part == "?"]


def render(source: str) -> str:
    operations = Counter(OP_RE.findall(source))
    dynamic_types = Counter(
        match for match in TENSOR_RE.findall(source) if "?" in match
    )
    revision = load_manifest()["revision"]
    lines = [
        "# E5 core operation and type inventory",
        "",
        "This file is generated from the generic-form textual core by",
        "`scripts/inventory_e5_core.py` and then manually reviewed. It records",
        f"the pinned model revision `{revision}`.",
        "Counts are textual occurrences in that deterministic bridge artifact,",
        "not runtime execution counts.",
        "",
        "Every `?` axis below is classified as the one entry sequence dimension",
        "`S`; every other axis is static. The pass contains the corresponding",
        "explicit allowlist and rejects any family not shown here.",
        "",
        "## Operations",
        "",
        "| Operation | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{name}` | {count} |" for name, count in sorted(operations.items())
    )
    lines.extend(
        [
            "",
            "## Dynamic tensor families",
            "",
            "| Tensor type | Occurrences | `S` axes |",
            "| --- | ---: | --- |",
        ]
    )
    for type_body, count in sorted(dynamic_types.items()):
        axes = ", ".join(str(axis) for axis in dynamic_axes(type_body))
        lines.append(f"| `tensor<{type_body}>` | {count} | {axes} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    actual = render(args.input.read_text(encoding="utf-8"))
    if args.check is None:
        print(actual, end="")
        return 0

    expected = args.check.read_text(encoding="utf-8")
    if actual != expected:
        diff = difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=str(args.check),
            tofile=str(args.input),
            lineterm="",
        )
        raise AssertionError("E5 core inventory changed:\n" + "\n".join(diff))
    print(f"PASS E5 core inventory: {args.check}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
