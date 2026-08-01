#!/usr/bin/env python3
"""Network-independent positive and rejection checks for E5 fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_e5_inputs import validate_inputs  # noqa: E402


def rejected(*values: np.ndarray) -> None:
    before = [value.copy() for value in values]
    try:
        validate_inputs(*values)
    except ValueError:
        if not all(np.array_equal(value, copy) for value, copy in zip(values, before)):
            raise AssertionError("input validator modified a rejected fixture")
        return
    raise AssertionError("invalid fixture was accepted")


def main() -> int:
    ids = np.arange(16, dtype=np.int64)[None, :]
    mask = np.ones_like(ids)
    types = np.zeros_like(ids)
    before = [value.copy() for value in (ids, mask, types)]
    validate_inputs(ids, mask, types)
    if not all(
        np.array_equal(value, copy) for value, copy in zip((ids, mask, types), before)
    ):
        raise AssertionError("input validator modified a valid fixture")
    rejected(ids, np.zeros_like(mask), types)
    rejected(ids, np.full_like(mask, 2), types)
    rejected(ids[:, :-1], mask, types)
    rejected(ids.astype(np.int32), mask, types)
    invalid_ids = ids.copy()
    invalid_ids[0, 0] = 30522
    rejected(invalid_ids, mask, types)
    invalid_types = types.copy()
    invalid_types[0, 0] = 2
    rejected(ids, mask, invalid_types)
    print("PASS E5 fixture validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
