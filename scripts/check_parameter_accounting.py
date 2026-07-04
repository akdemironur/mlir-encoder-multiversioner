#!/usr/bin/env python3
"""Check Stage A parameter-sharing invariants in transformed IR."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


CALL_RE = re.compile(r"func\.call @(?P<callee>mlp_s16|mlp_generic)\((?P<args>[^)]*)\)")
ARG_RE = re.compile(r"(%[\w$._-]+)\s*:")
WRAPPER_RE = re.compile(r"func\.func @mlp\((?P<args>[^)]*)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir-opt", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument(
        "--input", default=Path("examples/mlp/dynamic_mlp.mlir"), type=Path
    )
    return parser.parse_args()


def run_pass(args: argparse.Namespace) -> str:
    command = [
        str(args.mlir_opt),
        f"--load-pass-plugin={args.plugin}",
        "--pass-pipeline=builtin.module(shortseq-specialize)",
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


def check_parameter_forwarding(ir: str) -> None:
    wrapper_args, wrapper = parse_wrapper(ir)
    parameter_args = wrapper_args[1:5]
    calls = parse_call_args(wrapper)

    static_operands = calls.get("mlp_s16")
    if static_operands is None:
        raise AssertionError("@mlp does not call @mlp_s16")
    if static_operands[1:5] != parameter_args:
        raise AssertionError(
            "@mlp_s16 call does not forward wrapper parameter operands unchanged: "
            f"got {static_operands[1:5]}, expected {parameter_args}"
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
    ir = run_pass(args)

    check_no_dense_payloads(ir)
    check_parameter_forwarding(ir)

    print(
        "PASS parameter accounting: no generated dense payloads; "
        "@mlp forwards parameter operands to @mlp_s16 and @mlp_generic"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
