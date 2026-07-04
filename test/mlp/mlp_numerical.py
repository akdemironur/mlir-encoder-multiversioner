#!/usr/bin/env python3
"""Reference numerical checks for Stage A MLP dispatch."""

from __future__ import annotations

import math
import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Run with: "
        "uv run --frozen --group bench python test/mlp/mlp_numerical.py"
    ) from exc


HIDDEN = 384
INTERMEDIATE = 1536
STATIC_LENGTH = 16
FALLBACK_LENGTHS = (1, 7, 24)
RTOL = 2e-5
ATOL = 2e-5


def normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, shape).astype(np.float32)


def make_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0x5EED)
    return (
        normal(rng, (HIDDEN, INTERMEDIATE), 0.02),
        normal(rng, (INTERMEDIATE,), 0.01),
        normal(rng, (INTERMEDIATE, HIDDEN), 0.02),
        normal(rng, (HIDDEN,), 0.01),
    )


def make_input(sequence_length: int) -> np.ndarray:
    rng = np.random.default_rng(0xA11CE + sequence_length)
    return normal(rng, (1, sequence_length, HIDDEN), 0.1)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float32])
    scaled = x * np.float32(1.0 / math.sqrt(2.0))
    return (np.float32(0.5) * x * (np.float32(1.0) + erf(scaled))).astype(np.float32)


def mlp_body(
    x: np.ndarray, parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    w1, b1, w2, b2 = parameters
    sequence_length = x.shape[1]
    x_2d = x.reshape(sequence_length, HIDDEN)

    hidden = gelu_exact(x_2d @ w1 + b1)
    y_2d = hidden @ w2 + b2
    return (y_2d + x_2d).reshape(1, sequence_length, HIDDEN).astype(np.float32)


def mlp_generic(
    x: np.ndarray, parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    return mlp_body(x, parameters)


def mlp_s16(
    x: np.ndarray, parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    if x.shape[1] != STATIC_LENGTH:
        raise AssertionError(f"@mlp_s16 got S={x.shape[1]}, expected S=16")
    return mlp_body(x, parameters)


def mlp(
    x: np.ndarray, parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[str, np.ndarray]:
    if x.shape[1] == STATIC_LENGTH:
        return "@mlp_s16", mlp_s16(x, parameters)
    return "@mlp_generic", mlp_generic(x, parameters)


def assert_close(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=label)


def assert_path(actual: str, expected: str, sequence_length: int) -> None:
    if actual != expected:
        raise AssertionError(f"S={sequence_length}: selected {actual}, expected {expected}")


def main() -> int:
    parameters = make_parameters()

    x16 = make_input(STATIC_LENGTH)
    selected, wrapper_y = mlp(x16, parameters)
    assert_path(selected, "@mlp_s16", STATIC_LENGTH)
    assert_close("S=16 wrapper vs static", wrapper_y, mlp_s16(x16, parameters))
    assert_close("S=16 static vs generic", wrapper_y, mlp_generic(x16, parameters))

    for sequence_length in FALLBACK_LENGTHS:
        x = make_input(sequence_length)
        selected, wrapper_y = mlp(x, parameters)
        assert_path(selected, "@mlp_generic", sequence_length)
        assert_close(
            f"S={sequence_length} wrapper vs generic",
            wrapper_y,
            mlp_generic(x, parameters),
        )

    print("PASS Stage A MLP numerics: S=16 static path and non-16 fallback match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
