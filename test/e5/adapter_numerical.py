#!/usr/bin/env python3
"""Compare adapted ONNX pooling with encoder output plus NumPy reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from adapt_e5_onnx import validate_dynamic_contract  # noqa: E402
from e5_common import DEFAULT_ARTIFACT_DIR  # noqa: E402
from validate_e5_inputs import validate_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def numpy_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = mask.astype(np.float32)[..., None]
    mean = np.sum(hidden * expanded, axis=1) / np.sum(expanded, axis=1)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)


def hidden_output_name(path: Path) -> str:
    model = onnx.load(path, load_external_data=False)
    outputs = {value.name for value in model.graph.output}
    if "last_hidden_state" not in outputs:
        raise AssertionError("source model has no last_hidden_state output")
    return "last_hidden_state"


def fixture(length: int, padded: bool) -> dict[str, np.ndarray]:
    ids = ((np.arange(length, dtype=np.int64) * 37 + 101) % 30522)[None, :]
    mask = np.ones((1, length), dtype=np.int64)
    if padded:
        mask[:, length // 2 :] = 0
        ids[:, length // 2 :] = 0
    types = np.zeros((1, length), dtype=np.int64)
    validate_inputs(ids, mask, types)
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}


def check_other_dynamic_symbol_rejected(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    for value in model.graph.value_info:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "sequence_length":
                dim.dim_param = "other_dynamic"
                try:
                    validate_dynamic_contract(model)
                except AssertionError:
                    return
                raise AssertionError("adapter accepted a non-sequence dynamic axis")
    raise AssertionError("adapted model has no sequence-shaped intermediate")


def main() -> int:
    args = parse_args()
    source_path = args.artifact_dir / "model.onnx"
    adapted_path = args.artifact_dir / "sentence_embedding.onnx"
    source = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    adapted = ort.InferenceSession(
        str(adapted_path), providers=["CPUExecutionProvider"]
    )
    output_name = hidden_output_name(source_path)
    check_other_dynamic_symbol_rejected(adapted_path)
    for length, padded in ((17, False), (32, True)):
        inputs = fixture(length, padded)
        hidden = source.run([output_name], inputs)[0]
        expected = numpy_pool(hidden, inputs["attention_mask"])
        actual = adapted.run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        print(f"PASS S={length} padded={padded}")
    print("PASS residual dynamic-axis rejection")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
