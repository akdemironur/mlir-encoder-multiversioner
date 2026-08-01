#!/usr/bin/env python3
"""Check parameter-sharing invariants in transformed IR."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ARG_RE = re.compile(r"(%[\w$._-]+)\s*:")
PARAM_START = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir-opt", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument(
        "--input", default=Path("examples/mlp/dynamic_mlp.mlir"), type=Path
    )
    parser.add_argument("--lengths", default="16")
    parser.add_argument("--entry", default="mlp")
    parser.add_argument("--allow-unregistered-dialect", action="store_true")
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
        *(["--allow-unregistered-dialect"] if args.allow_unregistered_dialect else []),
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


def parse_wrapper(ir: str, entry: str) -> tuple[list[str], str]:
    wrapper_re = re.compile(rf"func\.func @{re.escape(entry)}\((?P<args>[^)]*)\)")
    match = wrapper_re.search(ir)
    if match is None:
        raise AssertionError(f"missing generated wrapper @{entry}")

    body_start = ir.find("{", match.end())
    if body_start == -1:
        raise AssertionError(f"could not find @{entry} wrapper body")

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
        raise AssertionError(f"unterminated @{entry} wrapper body")

    args = ARG_RE.findall(match.group("args"))
    if not args:
        raise AssertionError(f"expected @{entry} to have block arguments")
    return args, wrapper


def parse_call_args(wrapper: str, entry: str) -> dict[str, list[str]]:
    call_re = re.compile(
        rf"(?:func\.)?call @(?P<callee>{re.escape(entry)}_s\d+|"
        rf"{re.escape(entry)}_generic)\((?P<args>[^)]*)\)"
    )
    calls: dict[str, list[str]] = {}
    for match in call_re.finditer(wrapper):
        callee = match.group("callee")
        operands = [operand.strip() for operand in match.group("args").split(",")]
        calls[callee] = operands
    return calls


def check_no_dense_payloads(ir: str) -> None:
    if "dense<" in ir:
        raise AssertionError("transformed IR contains an embedded dense<...> payload")


def check_parameter_forwarding(ir: str, entry: str, lengths: list[int]) -> None:
    wrapper_args, wrapper = parse_wrapper(ir, entry)
    parameter_args = wrapper_args[PARAM_START:]
    calls = parse_call_args(wrapper, entry)

    for length in lengths:
        callee = f"{entry}_s{length}"
        static_operands = calls.get(callee)
        if static_operands is None:
            raise AssertionError(f"@{entry} does not call @{callee}")
        if static_operands[PARAM_START:] != parameter_args:
            raise AssertionError(
                f"@{callee} call does not forward wrapper parameter operands "
                f"unchanged: got {static_operands[PARAM_START:]}, "
                f"expected {parameter_args}"
            )

    generic_operands = calls.get(f"{entry}_generic")
    if generic_operands is None:
        raise AssertionError(f"@{entry} does not call @{entry}_generic")
    if generic_operands != wrapper_args:
        raise AssertionError(
            f"@{entry}_generic call does not forward wrapper operands unchanged: "
            f"got {generic_operands}, expected {wrapper_args}"
        )


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    ir = run_pass(args)

    check_no_dense_payloads(ir)
    check_parameter_forwarding(ir, args.entry, lengths)

    print(
        "PASS parameter accounting: no generated dense payloads; "
        f"@{args.entry} forwards parameter operands to static variants and fallback"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
