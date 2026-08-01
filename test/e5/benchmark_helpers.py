#!/usr/bin/env python3
"""Check the small parsers used by the E5 benchmark harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from bench_e5_iree import (  # noqa: E402
    parse_allocator_statistics,
    parse_module_metadata,
    parse_trials,
    parse_variants,
)


def main() -> int:
    assert parse_variants("dispatched,dynamic") == ["dynamic", "dispatched"]
    assert parse_module_metadata(
        "FlatBuffer: 120 bytes\n  Bytecode: 70 bytes\nExternal .rodata: ~40 bytes"
    ) == (120, 70, 40)
    assert parse_allocator_statistics(
        "HOST_LOCAL: 100B peak / 200B allocated\n"
        "DEVICE_LOCAL: 300B peak / 400B allocated"
    ) == (100, 300)
    benchmark_json = json.dumps(
        {
            "benchmarks": [
                {"run_type": "iteration", "real_time": 2500, "time_unit": "us"},
                {"run_type": "iteration", "real_time": 3, "time_unit": "ms"},
                {"run_type": "aggregate", "real_time": 2.75, "time_unit": "ms"},
            ]
        }
    )
    assert parse_trials(benchmark_json) == [2.5, 3.0]
    print("PASS E5 benchmark helper parsers")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
