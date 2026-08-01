#!/usr/bin/env python3
"""Compare ONNX, the dynamic VMFB, and every E5 dispatch path."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from e5_common import DEFAULT_ARTIFACT_DIR  # noqa: E402
from validate_e5_inputs import validate_inputs  # noqa: E402

F32_THRESHOLD = 0.0002
FIXTURES = (
    (16, "query", False),
    (32, "passage", True),
    (64, "query", True),
    (128, "passage", False),
    (17, "query", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def load_vocabulary(path: Path) -> dict[str, int]:
    tokens = path.read_text(encoding="utf-8").splitlines()
    vocabulary = {token: index for index, token in enumerate(tokens)}
    for required in (
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "query",
        "passage",
        ":",
        "hello",
        "world",
    ):
        if required not in vocabulary:
            raise AssertionError(f"vocabulary is missing {required!r}")
    return vocabulary


def make_fixture(
    length: int, prefix: str, padded: bool, vocabulary: dict[str, int]
) -> dict[str, np.ndarray]:
    active_length = length // 2 if padded else length
    if active_length < 5:
        raise AssertionError("fixture needs room for prefix and special tokens")

    content = [vocabulary["hello"], vocabulary["world"]]
    token_ids = [
        vocabulary["[CLS]"],
        vocabulary[prefix],
        vocabulary[":"],
    ]
    while len(token_ids) < active_length - 1:
        token_ids.append(content[(len(token_ids) - 3) % len(content)])
    token_ids.append(vocabulary["[SEP]"])
    token_ids.extend([vocabulary["[PAD]"]] * (length - active_length))

    ids = np.asarray(token_ids, dtype=np.int64)[None, :]
    mask = np.zeros((1, length), dtype=np.int64)
    mask[:, :active_length] = 1
    types = np.zeros((1, length), dtype=np.int64)
    validate_inputs(ids, mask, types)
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}


def run_iree(
    runner: str,
    module: Path,
    parameters: Path,
    inputs: dict[str, np.ndarray],
    expected: np.ndarray,
    temporary_directory: Path,
    label: str,
) -> None:
    paths: dict[str, Path] = {}
    arrays = {**inputs, "expected": expected}
    for name, array in arrays.items():
        path = temporary_directory / f"{label}-{name}.npy"
        np.save(path, array)
        paths[name] = path

    subprocess.run(
        [
            runner,
            f"--module={module}",
            "--device=local-task",
            f"--parameters=e5={parameters}",
            "--function=sentence_embedding",
            f"--input=@{paths['input_ids']}",
            f"--input=@{paths['attention_mask']}",
            f"--input=@{paths['token_type_ids']}",
            f"--expected_output=@{paths['expected']}",
            f"--expected_f32_threshold={F32_THRESHOLD}",
        ],
        check=True,
    )


def check_all_zero_mask_rejected(vocabulary: dict[str, int]) -> None:
    inputs = make_fixture(16, "query", False, vocabulary)
    inputs["attention_mask"].fill(0)
    originals = {name: value.copy() for name, value in inputs.items()}
    try:
        validate_inputs(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        )
    except ValueError as exc:
        if "at least one 1" not in str(exc):
            raise AssertionError(f"unexpected all-zero-mask diagnostic: {exc}") from exc
        for name, original in originals.items():
            np.testing.assert_array_equal(inputs[name], original)
        return
    raise AssertionError("all-zero attention mask was accepted")


def main() -> int:
    args = parse_args()
    runner = shutil.which("iree-run-module")
    if runner is None:
        raise AssertionError("required tool not found: iree-run-module")

    dynamic_module = args.artifact_dir / "sentence_embedding.dynamic.vmfb"
    dispatched_module = args.artifact_dir / "sentence_embedding.multiversioned.vmfb"
    parameters = args.artifact_dir / "e5.irpa"
    vocabulary = load_vocabulary(args.artifact_dir / "vocab.txt")
    session = ort.InferenceSession(
        str(args.artifact_dir / "sentence_embedding.onnx"),
        providers=["CPUExecutionProvider"],
    )

    with TemporaryDirectory(prefix="e5-full-numerical-") as temporary:
        temporary_directory = Path(temporary)
        for length, prefix, padded in FIXTURES:
            inputs = make_fixture(length, prefix, padded, vocabulary)
            expected = session.run(None, inputs)[0]
            label = f"s{length}-{prefix}-{'padded' if padded else 'unpadded'}"
            run_iree(
                runner,
                dynamic_module,
                parameters,
                inputs,
                expected,
                temporary_directory,
                f"{label}-dynamic",
            )
            run_iree(
                runner,
                dispatched_module,
                parameters,
                inputs,
                expected,
                temporary_directory,
                f"{label}-dispatched",
            )
            path = "generic" if length == 17 else f"static-s{length}"
            print(
                f"PASS S={length} path={path} prefix={prefix}: "
                f"padded={str(padded).lower()}"
            )

    check_all_zero_mask_rejected(vocabulary)
    print("PASS all-zero attention mask rejected before runtime invocation")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
