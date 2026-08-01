#!/usr/bin/env python3
"""Build the fixed batch-1 E5 sentence-embedding ONNX adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from e5_common import DEFAULT_ARTIFACT_DIR


INPUTS = ("input_ids", "attention_mask", "token_type_ids")
OUTPUT = "sentence_embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_ARTIFACT_DIR / "model.onnx"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "sentence_embedding.onnx",
    )
    return parser.parse_args()


def shape_of(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def get_hidden_output(model: onnx.ModelProto) -> onnx.ValueInfoProto:
    outputs = {value.name: value for value in model.graph.output}
    if "last_hidden_state" not in outputs:
        raise AssertionError("source model has no last_hidden_state output")
    hidden = outputs["last_hidden_state"]
    if len(shape_of(hidden)) != 3 or shape_of(hidden)[-1] != 384:
        raise AssertionError(f"unexpected last_hidden_state shape: {shape_of(hidden)}")
    return hidden


def assert_initializer_identity(
    before: dict[str, bytes], model: onnx.ModelProto
) -> None:
    after = {item.name: item.SerializeToString() for item in model.graph.initializer}
    if set(after) != set(before):
        raise AssertionError("adapter changed the source initializer key set")
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise AssertionError(f"adapter changed initializer bytes: {changed[:8]}")


def validate_dynamic_contract(model: onnx.ModelProto) -> None:
    input_values = {value.name: value for value in model.graph.input}
    sequence_symbols: set[str] = set()
    for name in INPUTS:
        shape = shape_of(input_values[name])
        if len(shape) != 2 or shape[0] != 1 or isinstance(shape[1], int):
            raise AssertionError(f"adapted {name} shape is {shape}, expected [1, S]")
        if isinstance(shape[1], str):
            sequence_symbols.add(shape[1])
    if len(sequence_symbols) != 1:
        raise AssertionError(f"input sequence dimensions disagree: {sequence_symbols}")
    sequence_symbol = next(iter(sequence_symbols))
    output = model.graph.output[0]
    if output.name != OUTPUT or shape_of(output) != [1, 384]:
        raise AssertionError(f"adapted output is {output.name} {shape_of(output)}")

    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        for dim in shape_of(value):
            if not isinstance(dim, int) and dim != sequence_symbol:
                raise AssertionError(
                    f"dynamic axis {dim!r} on {value.name} is not entry S"
                )


def main() -> int:
    args = parse_args()
    model = onnx.load(args.input, load_external_data=False)
    onnx.checker.check_model(model)
    original_initializers = {
        item.name: item.SerializeToString() for item in model.graph.initializer
    }
    graph_inputs = {value.name: value for value in model.graph.input}
    if set(graph_inputs) != set(INPUTS):
        raise AssertionError(f"unexpected ONNX inputs: {sorted(graph_inputs)}")
    for name in INPUTS:
        value = graph_inputs[name]
        if value.type.tensor_type.elem_type != TensorProto.INT64:
            raise AssertionError(f"{name} is not i64")
        dims = value.type.tensor_type.shape.dim
        if len(dims) != 2:
            raise AssertionError(f"{name} is not rank 2")
        dims[0].Clear()
        dims[0].dim_value = 1

    opset = next(
        (item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        None,
    )
    if opset != 11:
        raise AssertionError(f"expected canonical ONNX opset 11, got {opset}")
    hidden = get_hidden_output(model)
    graph = model.graph
    graph.name = "sentence_embedding"
    graph.node.extend(
        [
            helper.make_node(
                "Cast", ["attention_mask"], ["shortseq.mask_f32"], to=TensorProto.FLOAT
            ),
            helper.make_node(
                "Unsqueeze", ["shortseq.mask_f32"], ["shortseq.mask_expanded"], axes=[2]
            ),
            helper.make_node(
                "Mul",
                [hidden.name, "shortseq.mask_expanded"],
                ["shortseq.masked_hidden"],
            ),
            helper.make_node(
                "ReduceSum",
                ["shortseq.masked_hidden"],
                ["shortseq.masked_sum"],
                axes=[1],
                keepdims=0,
            ),
            helper.make_node(
                "ReduceSum",
                ["shortseq.mask_f32"],
                ["shortseq.mask_count"],
                axes=[1],
                keepdims=1,
            ),
            helper.make_node(
                "Div", ["shortseq.masked_sum", "shortseq.mask_count"], ["shortseq.mean"]
            ),
            helper.make_node(
                "ReduceL2",
                ["shortseq.mean"],
                ["shortseq.norm"],
                axes=[1],
                keepdims=1,
            ),
            helper.make_node("Div", ["shortseq.mean", "shortseq.norm"], [OUTPUT]),
        ]
    )
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, [1, 384])
    )

    inferred = SymbolicShapeInference.infer_shapes(
        model, auto_merge=True, guess_output_rank=False, verbose=0
    )
    onnx.checker.check_model(inferred, full_check=True)
    assert_initializer_identity(original_initializers, inferred)
    validate_dynamic_contract(inferred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, args.output)
    reloaded = onnx.load(args.output, load_external_data=False)
    assert_initializer_identity(original_initializers, reloaded)
    print(f"PASS wrote batch-1 sentence-embedding adapter: {args.output}")
    print(f"source_initializers={len(original_initializers)} unchanged")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
