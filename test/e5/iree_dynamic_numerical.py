#!/usr/bin/env python3
"""Compare the independently compiled IREE dynamic baseline with ONNX."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    length = 17
    ids = ((np.arange(length, dtype=np.int64) * 37 + 101) % 30522)[None, :]
    mask = np.ones((1, length), dtype=np.int64)
    types = np.zeros((1, length), dtype=np.int64)
    validate_inputs(ids, mask, types)

    onnx_session = ort.InferenceSession(
        str(args.artifact_dir / "sentence_embedding.onnx"),
        providers=["CPUExecutionProvider"],
    )
    expected = onnx_session.run(
        None,
        {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
    )[0]

    runner = shutil.which("iree-run-module")
    if runner is None:
        raise AssertionError("required tool not found: iree-run-module")
    with TemporaryDirectory(prefix="e5-dynamic-") as tmp:
        tmp_dir = Path(tmp)
        arrays = {
            "ids": ids,
            "mask": mask,
            "types": types,
            "expected": expected,
        }
        paths = {name: tmp_dir / f"{name}.npy" for name in arrays}
        for name, array in arrays.items():
            np.save(paths[name], array)
        subprocess.run(
            [
                runner,
                f"--module={args.artifact_dir / 'sentence_embedding.dynamic.vmfb'}",
                "--device=local-task",
                f"--parameters=e5={args.artifact_dir / 'e5.irpa'}",
                "--function=sentence_embedding",
                f"--input=@{paths['ids']}",
                f"--input=@{paths['mask']}",
                f"--input=@{paths['types']}",
                f"--expected_output=@{paths['expected']}",
                "--expected_f32_threshold=0.0002",
            ],
            check=True,
        )

    print("PASS IREE dynamic E5 baseline matches ONNX at S=17")
    print(
        f"vmfb_bytes={(args.artifact_dir / 'sentence_embedding.dynamic.vmfb').stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
