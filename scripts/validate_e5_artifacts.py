#!/usr/bin/env python3
"""Validate the pinned official E5 ONNX and tokenizer metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import onnx
from onnx import TensorProto

from e5_common import DEFAULT_ARTIFACT_DIR, validate_hashes


INPUTS = ("input_ids", "attention_mask", "token_type_ids")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    dims: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def main() -> int:
    args = parse_args()
    validate_hashes(args.artifact_dir)
    with (args.artifact_dir / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    expected_config = {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
        "type_vocab_size": 2,
        "torch_dtype": "float32",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise AssertionError(
                f"config {key}={config.get(key)!r}, expected {expected!r}"
            )
    vocab_lines = (
        (args.artifact_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    )
    if len(vocab_lines) != 30522:
        raise AssertionError(
            f"vocabulary has {len(vocab_lines)} entries, expected 30522"
        )

    model = onnx.load(args.artifact_dir / "model.onnx", load_external_data=False)
    onnx.checker.check_model(model)
    inputs = {value.name: value for value in model.graph.input}
    if set(inputs) != set(INPUTS):
        raise AssertionError(
            f"ONNX inputs are {sorted(inputs)}, expected {list(INPUTS)}"
        )
    for name in INPUTS:
        value = inputs[name]
        if value.type.tensor_type.elem_type != TensorProto.INT64:
            raise AssertionError(f"{name} must be i64")
        shape = tensor_shape(value)
        if len(shape) != 2 or isinstance(shape[0], int) or isinstance(shape[1], int):
            raise AssertionError(
                f"{name} must have two dynamic dimensions, got {shape}"
            )

    if not model.graph.initializer:
        raise AssertionError("ONNX graph has no initializers")
    non_f32 = {
        item.name: (item.data_type, tuple(item.dims))
        for item in model.graph.initializer
        if item.data_type != TensorProto.FLOAT
    }
    expected_metadata = {"embeddings.position_ids": (TensorProto.INT64, (1, 512))}
    if non_f32 != expected_metadata:
        raise AssertionError(
            f"non-f32 initializers are {non_f32}, expected only position metadata"
        )
    layer_numbers = {
        int(match.group(1))
        for item in model.graph.initializer
        if (match := re.search(r"encoder\.layer\.(\d+)\.", item.name))
    }
    if layer_numbers != set(range(12)):
        raise AssertionError(
            f"encoder layer initializers cover {sorted(layer_numbers)}"
        )
    hidden_outputs = [
        value for value in model.graph.output if tensor_shape(value)[-1:] == [384]
    ]
    if not hidden_outputs:
        raise AssertionError("no ONNX output has hidden dimension 384")

    parameter_bytes = sum(
        len(item.SerializeToString()) for item in model.graph.initializer
    )
    print(f"PASS pinned E5 artifacts: {args.artifact_dir}")
    print(f"initializers={len(model.graph.initializer)}")
    print(f"serialized_initializer_bytes={parameter_bytes}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
