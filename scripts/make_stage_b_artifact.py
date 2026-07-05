#!/usr/bin/env python3
"""Write synthetic Stage B params and sample inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Run with: "
        "uv run --frozen --group numerical python scripts/make_stage_b_artifact.py"
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
        "--output",
        default=Path("results/stage_b/encoder_block_synthetic.npz"),
        type=Path,
    )
    parser.add_argument("--lengths", default="4,8,16")
    parser.add_argument("--seed", default=0xC0FFEE, type=int)
    return parser.parse_args()


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item) for item in text.split(",") if item]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must be positive integers")
    return lengths


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    rng = np.random.default_rng(args.seed)

    arrays = {
        name: rng.normal(0.0, 0.02, shape).astype(np.float32)
        for name, shape in PARAMS.items()
    }
    arrays.update(
        {
            f"x_s{length}": rng.normal(0.0, 0.1, (1, length, HIDDEN)).astype(np.float32)
            for length in lengths
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)

    print(f"wrote {args.output}")
    for name, array in arrays.items():
        print(f"{name}: shape={array.shape} dtype={array.dtype} bytes={array.nbytes}")
    param_bytes = sum(arrays[name].nbytes for name in PARAMS)
    input_bytes = sum(arrays[f"x_s{length}"].nbytes for length in lengths)
    print(f"parameter_bytes={param_bytes}")
    print(f"sample_input_bytes={input_bytes}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
