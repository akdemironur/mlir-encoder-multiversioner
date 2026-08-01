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
SSA = r"%[-A-Za-z0-9_.$]+"


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


def extract_braced_region(text: str, opening_brace: int) -> tuple[str, int]:
    depth = 0
    for offset in range(opening_brace, len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace + 1 : offset], offset + 1
    raise AssertionError("unterminated dispatch region")


def split_if_regions(text: str, if_match: re.Match[str]) -> tuple[str, str]:
    then_open = text.find("{", if_match.end())
    if then_open == -1:
        raise AssertionError("dispatch scf.if has no then region")
    then_body, after_then = extract_braced_region(text, then_open)
    else_match = re.match(r"\s*else\s*", text[after_then:])
    if else_match is None:
        raise AssertionError("dispatch scf.if has no else region")
    else_open = text.find("{", after_then + else_match.end())
    if else_open == -1:
        raise AssertionError("dispatch scf.if has no else body")
    else_body, _ = extract_braced_region(text, else_open)
    return then_body, else_body


def one_match(pattern: str, text: str, description: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise AssertionError(f"expected one {description}, found {len(matches)}")
    return matches[0]


def check_dispatch_wrapper(core: str, lengths: list[int]) -> None:
    wrapper = extract_function(core, "sentence_embedding")
    header = wrapper[: wrapper.find("{")]
    arguments = re.findall(rf"({SSA}): tensor<1x\?xi64>", header)
    if len(arguments) != 3:
        raise AssertionError("E5 dispatch wrapper must have three dynamic token inputs")

    dimensions = []
    axes = []
    for argument in arguments:
        match = one_match(
            rf"({SSA}) = tensor\.dim {re.escape(argument)}, ({SSA}) "
            r": tensor<1x\?xi64>",
            wrapper,
            f"sequence dimension for {argument}",
        )
        dimensions.append(match.group(1))
        axes.append(match.group(2))
    if len(set(axes)) != 1 or not re.search(
        rf"{re.escape(axes[0])} = arith\.constant 1 : index", wrapper
    ):
        raise AssertionError("E5 dispatch dimensions must all read axis 1")

    expected_calls = [f"sentence_embedding_s{length}" for length in lengths]
    expected_calls.append("sentence_embedding_generic")
    actual_calls = re.findall(r"func\.call @([A-Za-z0-9_.$-]+)\(", wrapper)
    if actual_calls != expected_calls:
        raise AssertionError(
            f"E5 wrapper calls {actual_calls}, expected {expected_calls}"
        )

    remaining = wrapper
    outer_result = None
    for index, length in enumerate(lengths):
        constant = one_match(
            rf"({SSA}) = arith\.constant {length} : index",
            remaining,
            f"S={length} dispatch constant",
        ).group(1)
        comparisons = []
        for dimension in dimensions:
            comparison = one_match(
                rf"({SSA}) = arith\.cmpi eq, {re.escape(dimension)}, "
                rf"{re.escape(constant)} : index",
                remaining,
                f"S={length} equality for {dimension}",
            ).group(1)
            comparisons.append(comparison)
        first_and = one_match(
            rf"({SSA}) = arith\.andi {re.escape(comparisons[0])}, "
            rf"{re.escape(comparisons[1])} : i1",
            remaining,
            f"S={length} first guard conjunction",
        ).group(1)
        full_guard = one_match(
            rf"({SSA}) = arith\.andi {re.escape(first_and)}, "
            rf"{re.escape(comparisons[2])} : i1",
            remaining,
            f"S={length} three-input guard",
        ).group(1)
        branch = one_match(
            rf"({SSA}) = scf\.if {re.escape(full_guard)} "
            r"-> \(tensor<1x384xf32>\)",
            remaining,
            f"S={length} dispatch branch",
        )
        if index == 0:
            outer_result = branch.group(1)
        then_body, remaining = split_if_regions(remaining, branch)
        calls = re.findall(r"func\.call @([A-Za-z0-9_.$-]+)\(", then_body)
        expected_callee = f"sentence_embedding_s{length}"
        if calls != [expected_callee]:
            raise AssertionError(
                f"S={length} branch calls {calls}, expected @{expected_callee}"
            )
        static_result = one_match(
            rf"({SSA}) = func\.call @{expected_callee}\(",
            then_body,
            f"S={length} static call",
        ).group(1)
        one_match(
            rf"scf\.yield {re.escape(static_result)} : tensor<1x384xf32>",
            then_body,
            f"S={length} yielded static result",
        )

    if "scf.if" in remaining:
        raise AssertionError("E5 final fallback contains an unexpected branch")
    calls = re.findall(r"func\.call @([A-Za-z0-9_.$-]+)\(", remaining)
    if calls != ["sentence_embedding_generic"]:
        raise AssertionError(
            "E5 final fallback must call @sentence_embedding_generic exactly once"
        )
    generic_call = one_match(
        rf"({SSA}) = func\.call @sentence_embedding_generic\(([^)]*)\)",
        remaining,
        "generic fallback call",
    )
    operands = [item.strip() for item in generic_call.group(2).split(",")]
    if operands != arguments:
        raise AssertionError("E5 generic fallback must receive the original inputs")
    one_match(
        rf"scf\.yield {re.escape(generic_call.group(1))} : tensor<1x384xf32>",
        remaining,
        "yielded generic fallback result",
    )
    one_match(
        rf"return {re.escape(outer_result)} : tensor<1x384xf32>",
        wrapper,
        "returned outer dispatch result",
    )


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
        check_dispatch_wrapper(core, lengths)

    canonical_bytes = sum(entry.length for entry in entries.values())
    print("PASS E5 bridge parameter identity and IR structure")
    if lengths:
        print("static_sequence_lengths=" + ",".join(map(str, lengths)))
    print(f"irpa_entries={len(entries)}")
    print(f"canonical_parameter_bytes={canonical_bytes}")
    print("duplicated_parameter_bytes=0")
    print("length_specific_parameter_bytes=0")
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
