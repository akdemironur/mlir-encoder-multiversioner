#!/usr/bin/env python3
"""Reference numerical checks for the Stage B encoder-block dispatch."""

from __future__ import annotations

import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Run with: "
        "uv run --frozen --group numerical python test/stage_b/encoder_block_numerical.py"
    ) from exc


HIDDEN = 64
INTERMEDIATE = 256
STATIC_LENGTHS = (4, 8)
FALLBACK_LENGTHS = (1, 7, 16)
RTOL = 2e-5
ATOL = 2e-5


Params = tuple[np.ndarray, ...]


def normal(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float
) -> np.ndarray:
    return rng.normal(0.0, scale, shape).astype(np.float32)


def make_parameters() -> Params:
    rng = np.random.default_rng(0xC0FFEE)
    return (
        normal(rng, (HIDDEN, HIDDEN), 0.02),
        normal(rng, (HIDDEN, HIDDEN), 0.02),
        normal(rng, (HIDDEN, HIDDEN), 0.02),
        normal(rng, (HIDDEN, HIDDEN), 0.02),
        normal(rng, (HIDDEN,), 0.01),
        normal(rng, (HIDDEN,), 0.01),
        normal(rng, (HIDDEN, INTERMEDIATE), 0.02),
        normal(rng, (INTERMEDIATE,), 0.01),
        normal(rng, (INTERMEDIATE, HIDDEN), 0.02),
        normal(rng, (HIDDEN,), 0.01),
    )


def make_input(sequence_length: int) -> np.ndarray:
    rng = np.random.default_rng(0xB10C + sequence_length)
    return normal(rng, (1, sequence_length, HIDDEN), 0.1)


def encoder_block_body(x: np.ndarray, parameters: Params) -> np.ndarray:
    q_w, k_w, v_w, o_w, norm_scale, norm_bias, ff_w1, ff_b1, ff_w2, ff_b2 = parameters
    sequence_length = x.shape[1]
    x_2d = x.reshape(sequence_length, HIDDEN)

    norm0 = x_2d * norm_scale + norm_bias
    q = norm0 @ q_w
    k = norm0 @ k_w
    v = norm0 @ v_w
    attn = (q + k + v) @ o_w
    residual0 = attn + x_2d
    norm1 = residual0 * norm_scale + norm_bias
    ff1_biased = norm1 @ ff_w1 + ff_b1
    ff1 = ff1_biased * ff1_biased
    ff2 = ff1 @ ff_w2
    out = ff2 + ff_b2 + residual0
    return out.reshape(1, sequence_length, HIDDEN).astype(np.float32)


def encoder_block_generic(x: np.ndarray, parameters: Params) -> np.ndarray:
    return encoder_block_body(x, parameters)


def encoder_block_static(
    x: np.ndarray, parameters: Params, sequence_length: int
) -> np.ndarray:
    if x.shape[1] != sequence_length:
        raise AssertionError(
            f"@encoder_block_s{sequence_length} got S={x.shape[1]}, "
            f"expected S={sequence_length}"
        )
    return encoder_block_body(x, parameters)


def encoder_block(x: np.ndarray, parameters: Params) -> tuple[str, np.ndarray]:
    parameter_ids = tuple(id(parameter) for parameter in parameters)
    sequence_length = x.shape[1]
    if sequence_length in STATIC_LENGTHS:
        result = encoder_block_static(x, parameters, sequence_length)
        selected = f"@encoder_block_s{sequence_length}"
    else:
        result = encoder_block_generic(x, parameters)
        selected = "@encoder_block_generic"

    if tuple(id(parameter) for parameter in parameters) != parameter_ids:
        raise AssertionError("@encoder_block did not preserve parameter references")
    return selected, result


def assert_close(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=label)


def assert_path(actual: str, expected: str, sequence_length: int) -> None:
    if actual != expected:
        raise AssertionError(
            f"S={sequence_length}: selected {actual}, expected {expected}"
        )


def main() -> int:
    parameters = make_parameters()

    for sequence_length in STATIC_LENGTHS:
        x = make_input(sequence_length)
        selected, wrapper_y = encoder_block(x, parameters)
        assert_path(selected, f"@encoder_block_s{sequence_length}", sequence_length)
        assert_close(
            f"S={sequence_length} wrapper vs static",
            wrapper_y,
            encoder_block_static(x, parameters, sequence_length),
        )
        assert_close(
            f"S={sequence_length} static vs generic",
            wrapper_y,
            encoder_block_generic(x, parameters),
        )

    for sequence_length in FALLBACK_LENGTHS:
        x = make_input(sequence_length)
        selected, wrapper_y = encoder_block(x, parameters)
        assert_path(selected, "@encoder_block_generic", sequence_length)
        assert_close(
            f"S={sequence_length} wrapper vs generic",
            wrapper_y,
            encoder_block_generic(x, parameters),
        )

    length_text = ",".join(str(length) for length in STATIC_LENGTHS)
    print(
        "PASS Stage B encoder-block numerics: "
        f"static paths S={length_text} and generic fallback match"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
