#!/usr/bin/env python3
"""Check E5 external-parameter identity and textual core invariants."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import iree.runtime as ireert
import onnx

from e5_common import DEFAULT_ARTIFACT_DIR


REFERENCE_RE = re.compile(r'#stream\.parameter\.named<"e5"::"([^"]+)">')
LENGTH_SUFFIX_RE = re.compile(r"_s(?:16|32|64|128)(?:\b|_)")
INLINE_DENSE_RE = re.compile(
    r"\bdense<.*?>\s*:\s*tensor<([^>]+)>", re.DOTALL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def irpa_key(onnx_name: str) -> str:
    # MLIR symbols cannot contain the ONNX exporter delimiter `::`.
    return onnx_name.replace("::", "__")


def tensor_element_count(type_body: str) -> int:
    # The audited core uses ordinary builtin tensor types without encodings.
    shape_and_element = type_body.split(",", 1)[0].strip()
    parts = shape_and_element.split("x")
    if len(parts) == 1:
        return 1
    try:
        dimensions = [int(part) for part in parts[:-1]]
    except ValueError as exc:
        raise AssertionError(
            f"cannot account for inline dense tensor type tensor<{type_body}>"
        ) from exc
    count = 1
    for dimension in dimensions:
        count *= dimension
    return count


def check_no_dense_payloads(core: str) -> None:
    if "dense_resource<" in core:
        raise AssertionError("core MLIR contains an embedded dense resource")
    for type_body in INLINE_DENSE_RE.findall(core):
        elements = tensor_element_count(type_body)
        if elements > 1:
            raise AssertionError(
                "core MLIR contains a non-scalar inline dense tensor: "
                f"tensor<{type_body}> has {elements} elements"
            )


def main() -> int:
    args = parse_args()
    model = onnx.load(
        args.artifact_dir / "sentence_embedding.onnx", load_external_data=False
    )
    initializers = {irpa_key(item.name): item for item in model.graph.initializer}
    if len(initializers) != len(model.graph.initializer):
        raise AssertionError("ONNX-to-IRPA key normalization has a collision")

    index = ireert.ParameterIndex()
    index.load(str(args.artifact_dir / "e5.irpa"))
    entries = {key: entry for key, entry in index.items()}
    if set(entries) != set(initializers):
        raise AssertionError("IRPA keys do not match canonical ONNX initializers")
    for key, initializer in initializers.items():
        if not initializer.raw_data:
            raise AssertionError(f"canonical initializer has no raw_data: {key}")
        expected = initializer.raw_data
        actual = bytes(entries[key].file_view)
        if actual != expected:
            raise AssertionError(f"IRPA bytes differ for {key}")

    core = (args.artifact_dir / "sentence_embedding.core.mlir").read_text(
        encoding="utf-8"
    )
    references = REFERENCE_RE.findall(core)
    if set(references) != set(initializers) or len(references) != len(initializers):
        raise AssertionError("core MLIR must reference each canonical parameter once")
    if core.count('"util.global"') != len(initializers):
        raise AssertionError("core MLIR must contain one global per initializer")
    check_no_dense_payloads(core)
    if LENGTH_SUFFIX_RE.search(core) or any(
        LENGTH_SUFFIX_RE.search(key) for key in entries
    ):
        raise AssertionError("found a length-specific parameter copy")
    if core.count("shortseq.e5_small_v2") != 1 or core.count("shortseq.entry") != 1:
        raise AssertionError("core MLIR must contain one marked E5 entry")

    canonical_bytes = sum(entry.length for entry in entries.values())
    print("PASS E5 bridge parameter identity")
    print(f"canonical_parameter_bytes={canonical_bytes}")
    print("duplicated_parameter_bytes=0")
    print("prepacked_weight_bytes=not_created_at_core_boundary")
    print("variant_metadata_bytes=0")
    print("active_scratch_bytes=not_allocated_at_core_boundary")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
