#!/usr/bin/env python3
"""Check a synthetic Stage B artifact."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Run with: "
        "uv run --frozen --group numerical python scripts/check_stage_b_artifact.py"
    ) from exc


HIDDEN = 64
INTERMEDIATE = 256
PARAMS = {
    "q_w": (HIDDEN, HIDDEN),
    "k_w": (HIDDEN, HIDDEN),
    "v_w": (HIDDEN, HIDDEN),
    "o_w": (HIDDEN, HIDDEN),
    "norm_scale": (HIDDEN,),
    "norm_bias": (HIDDEN,),
    "ff_w1": (HIDDEN, INTERMEDIATE),
    "ff_b1": (INTERMEDIATE,),
    "ff_w2": (INTERMEDIATE, HIDDEN),
    "ff_b2": (HIDDEN,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=Path("results/stage_b/encoder_block_synthetic.npz"),
        type=Path,
    )
    return parser.parse_args()


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_array(data: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]) -> int:
    expect(name in data.files, f"missing {name}")
    array = data[name]
    expect(array.shape == shape, f"{name} shape {array.shape}, expected {shape}")
    expect(array.dtype == np.float32, f"{name} dtype {array.dtype}, expected float32")
    return int(array.nbytes)


def main() -> int:
    args = parse_args()
    with np.load(args.input) as data:
        param_bytes = sum(check_array(data, name, shape) for name, shape in PARAMS.items())
        sample_keys = sorted(
            (key for key in data.files if re.fullmatch(r"x_s[1-9][0-9]*", key)),
            key=lambda name: int(name[3:]),
        )
        expect(sample_keys, "missing sample inputs x_sN")
        sample_bytes = 0
        for key in sample_keys:
            length = int(key[3:])
            sample_bytes += check_array(data, key, (1, length, HIDDEN))

        extra_params = [
            key
            for key in data.files
            if re.fullmatch(r".*_s[1-9][0-9]*", key) and not key.startswith("x_s")
        ]
        expect(not extra_params, f"length-specific parameter keys: {extra_params}")

    print(f"PASS Stage B artifact: {args.input}")
    print(f"parameter_bytes={param_bytes}")
    print(f"sample_input_bytes={sample_bytes}")
    print(f"sample_inputs={','.join(sample_keys)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
