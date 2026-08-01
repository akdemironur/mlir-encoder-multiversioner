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
    parser.add_argument("--core", type=Path)
    parser.add_argument("--specialized-lengths", default="")
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


def extract_function(core: str, name: str) -> str:
    match = re.search(rf"\bfunc\.func(?: private)? @{re.escape(name)}\(", core)
    if match is None:
        raise AssertionError(f"missing function @{name}")
    body_start = core.find("{", match.end())
    if body_start == -1:
        raise AssertionError(f"missing body for @{name}")
    depth = 0
    for offset, character in enumerate(core[body_start:], start=body_start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return core[match.start() : offset + 1]
    raise AssertionError(f"unterminated body for @{name}")


def parse_lengths(text: str) -> list[int]:
    if not text:
        return []
    lengths = [int(item) for item in text.split(",")]
    if any(length <= 0 for length in lengths) or len(lengths) != len(set(lengths)):
        raise AssertionError("--specialized-lengths must be unique positive integers")
    return sorted(lengths)


def check_static_clones(
    core: str, lengths: list[int], initializer_count: int
) -> None:
    generic = extract_function(core, "sentence_embedding_generic")
    if generic.count('"util.global.load"') != initializer_count:
        raise AssertionError("generic E5 function must load every shared parameter")

    for length in lengths:
        clone = extract_function(core, f"sentence_embedding_s{length}")
        if "?" in clone:
            raise AssertionError(f"E5 S={length} clone retains a dynamic dimension")
        if clone.count(f"tensor<1x{length}xi64>") < 3:
            raise AssertionError(f"E5 S={length} clone has the wrong token ABI")
        if "-> tensor<1x384xf32>" not in clone:
            raise AssertionError(f"E5 S={length} clone has the wrong result type")
        if clone.count('"util.global.load"') != initializer_count:
            raise AssertionError(
                f"E5 S={length} clone must load every shared parameter"
            )


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.specialized_lengths)
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

    core_path = args.core or args.artifact_dir / "sentence_embedding.core.mlir"
    core = core_path.read_text(encoding="utf-8")
    references = REFERENCE_RE.findall(core)
    if set(references) != set(initializers) or len(references) != len(initializers):
        raise AssertionError("core MLIR must reference each canonical parameter once")
    if core.count('"util.global"') != len(initializers):
        raise AssertionError("core MLIR must contain one global per initializer")
    check_no_dense_payloads(core)
    if any(LENGTH_SUFFIX_RE.search(key) for key in entries) or any(
        LENGTH_SUFFIX_RE.search(reference) for reference in references
    ):
        raise AssertionError("found a length-specific parameter copy")
    expected_markers = 0 if lengths else 1
    if (
        core.count("shortseq.e5_small_v2") != expected_markers
        or core.count("shortseq.entry") != expected_markers
    ):
        raise AssertionError("core MLIR has the wrong number of E5 entry markers")
    if lengths:
        check_static_clones(core, lengths, len(initializers))

    canonical_bytes = sum(entry.length for entry in entries.values())
    print("PASS E5 bridge parameter identity and IR structure")
    if lengths:
        print("static_sequence_lengths=" + ",".join(map(str, lengths)))
    print(f"canonical_parameter_bytes={canonical_bytes}")
    print("duplicated_parameter_bytes=0")
    print("prepacked_weight_bytes=not_created_at_core_boundary")
    if lengths:
        print("variant_metadata_bytes=not_serialized_at_textual_core_boundary")
    else:
        print("variant_metadata_bytes=0")
    print("active_scratch_bytes=not_allocated_at_core_boundary")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
