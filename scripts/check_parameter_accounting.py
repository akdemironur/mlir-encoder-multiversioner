#!/usr/bin/env python3
"""Check Stage A parameter-sharing invariants in transformed IR."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


CALL_RE = re.compile(
    r"func\.call @(?P<callee>mlp_s\d+|mlp_generic)\((?P<args>[^)]*)\)"
)
ARG_RE = re.compile(r"(%[\w$._-]+)\s*:")
WRAPPER_RE = re.compile(r"func\.func @mlp\((?P<args>[^)]*)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir-opt", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument(
        "--input", default=Path("examples/mlp/dynamic_mlp.mlir"), type=Path
    )
    parser.add_argument("--lengths", default="16")
    return parser.parse_args()


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive integers")
    if len(lengths) != len(set(lengths)):
        raise ValueError("--lengths must not contain duplicates")
    return sorted(lengths)


def run_pass(args: argparse.Namespace) -> str:
    command = [
        str(args.mlir_opt),
        f"--load-pass-plugin={args.plugin}",
        "--pass-pipeline="
        f"builtin.module(shortseq-specialize{{lengths={args.lengths}}})",
        str(args.input),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    return completed.stdout


def parse_wrapper(ir: str) -> tuple[list[str], str]:
    match = WRAPPER_RE.search(ir)
    if match is None:
        raise AssertionError("missing generated wrapper @mlp")

    body_start = ir.find("{", match.end())
    if body_start == -1:
        raise AssertionError("could not find @mlp wrapper body")

    depth = 0
    for offset, char in enumerate(ir[body_start:], start=body_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                wrapper = ir[match.start() : offset + 1]
                break
    else:
        raise AssertionError("unterminated @mlp wrapper body")

    args = ARG_RE.findall(match.group("args"))
    if len(args) != 5:
        raise AssertionError(f"expected @mlp to have 5 block arguments, got {args}")
    return args, wrapper


def parse_call_args(wrapper: str) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {}
    for match in CALL_RE.finditer(wrapper):
        callee = match.group("callee")
        operands = [operand.strip() for operand in match.group("args").split(",")]
        calls[callee] = operands
    return calls


def check_no_dense_payloads(ir: str) -> None:
    if "dense<" in ir:
        raise AssertionError("transformed IR contains an embedded dense<...> payload")


def check_parameter_forwarding(ir: str, lengths: list[int]) -> None:
    wrapper_args, wrapper = parse_wrapper(ir)
    parameter_args = wrapper_args[1:5]
    calls = parse_call_args(wrapper)

    for length in lengths:
        callee = f"mlp_s{length}"
        static_operands = calls.get(callee)
        if static_operands is None:
            raise AssertionError(f"@mlp does not call @{callee}")
        if static_operands[1:5] != parameter_args:
            raise AssertionError(
                f"@{callee} call does not forward wrapper parameter operands "
                f"unchanged: got {static_operands[1:5]}, expected {parameter_args}"
            )

    generic_operands = calls.get("mlp_generic")
    if generic_operands is None:
        raise AssertionError("@mlp does not call @mlp_generic")
    if generic_operands != wrapper_args:
        raise AssertionError(
            "@mlp_generic call does not forward wrapper operands unchanged: "
            f"got {generic_operands}, expected {wrapper_args}"
        )


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    ir = run_pass(args)

    check_no_dense_payloads(ir)
    check_parameter_forwarding(ir, lengths)

    print(
        "PASS parameter accounting: no generated dense payloads; "
        "@mlp forwards parameter operands to static variants and @mlp_generic"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
