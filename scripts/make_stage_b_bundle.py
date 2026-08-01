#!/usr/bin/env python3
"""Build a Stage B MLIR module plus one canonical IREE parameter archive."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import iree.runtime as ireert
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "IREE runtime and NumPy are required. Run with: "
        "uv run --frozen --group bench python scripts/make_stage_b_bundle.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = (
    ("q_w", (64, 64), "tensor<64x64xf32>"),
    ("k_w", (64, 64), "tensor<64x64xf32>"),
    ("v_w", (64, 64), "tensor<64x64xf32>"),
    ("o_w", (64, 64), "tensor<64x64xf32>"),
    ("norm_scale", (64,), "tensor<64xf32>"),
    ("norm_bias", (64,), "tensor<64xf32>"),
    ("ff_w1", (64, 256), "tensor<64x256xf32>"),
    ("ff_b1", (256,), "tensor<256xf32>"),
    ("ff_w2", (256, 64), "tensor<256x64xf32>"),
    ("ff_b2", (64,), "tensor<64xf32>"),
)
SCOPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default=REPO_ROOT / "results/stage_b/encoder_block_synthetic.npz",
        type=Path,
    )
    parser.add_argument(
        "--source",
        default=REPO_ROOT / "examples/stage_b/contract_encoder_block.mlir",
        type=Path,
    )
    parser.add_argument(
        "--output-mlir",
        default=REPO_ROOT / "results/stage_b/encoder_block_bundle.mlir",
        type=Path,
    )
    parser.add_argument(
        "--output-irpa",
        default=REPO_ROOT / "results/stage_b/encoder_block_weights.irpa",
        type=Path,
    )
    parser.add_argument("--scope", default="stage_b")
    return parser.parse_args()


def load_parameters(artifact: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(artifact) as data:
        for name, shape, _ in PARAMS:
            if name not in data.files:
                raise AssertionError(f"missing parameter {name}")
            array = data[name]
            if array.shape != shape:
                raise AssertionError(f"{name} shape {array.shape}, expected {shape}")
            if array.dtype != np.float32:
                raise AssertionError(
                    f"{name} dtype {array.dtype}, expected float32"
                )
            arrays[name] = np.ascontiguousarray(array)

        copied_params = [
            key
            for key in data.files
            if re.fullmatch(r".*_s[1-9][0-9]*", key) and not key.startswith("x_s")
        ]
        if copied_params:
            raise AssertionError(f"length-specific parameter keys: {copied_params}")
    return arrays


def source_body(source: Path) -> str:
    match = re.fullmatch(r"\s*module\s*\{(?P<body>.*)\}\s*", source.read_text(), re.DOTALL)
    if match is None:
        raise ValueError(f"{source} must contain exactly one top-level module")
    return match.group("body").strip()


def render_mlir(source: Path, scope: str) -> str:
    if SCOPE_RE.fullmatch(scope) is None:
        raise ValueError(
            "--scope must start with a letter or underscore and contain only "
            "letters, digits, '_', '.', or '-'"
        )

    globals_ir = []
    loads_ir = []
    call_operands = ["%x"]
    call_types = ["tensor<1x?x64xf32>"]
    for name, _, mlir_type in PARAMS:
        symbol = f"stage_b_{name}"
        globals_ir.append(
            f'  "util.global"() <{{initial_value = '
            f'#flow.parameter.named<"{scope}"::"{name}"> : {mlir_type}, '
            f'sym_name = "{symbol}", sym_visibility = "private", '
            f"type = {mlir_type}}}> : () -> ()"
        )
        loads_ir.append(
            f'    %{name} = "util.global.load"() <{{global = @{symbol}}}> '
            f": () -> {mlir_type}"
        )
        call_operands.append(f"%{name}")
        call_types.append(mlir_type)

    call_operands_text = ", ".join(call_operands)
    call_types_text = ", ".join(call_types)
    indented_body = "\n".join(f"  {line}" for line in source_body(source).splitlines())
    return (
        "module {\n"
        + "\n".join(globals_ir)
        + "\n\n"
        + "  func.func @run_encoder_block(\n"
        + "      %x: tensor<1x?x64xf32>\n"
        + "  ) -> tensor<1x?x64xf32> {\n"
        + "\n".join(loads_ir)
        + "\n"
        + f"    %y = func.call @encoder_block({call_operands_text})\n"
        + f"        : ({call_types_text}) -> tensor<1x?x64xf32>\n"
        + "    return %y : tensor<1x?x64xf32>\n"
        + "  }\n\n"
        + indented_body
        + "\n}\n"
    )


def write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    index = ireert.ParameterIndex()
    for name, _, _ in PARAMS:
        index.add_buffer(name, arrays[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    index.create_archive_file(str(path))


def build_bundle(
    artifact: Path, source: Path, output_mlir: Path, output_irpa: Path, scope: str
) -> int:
    arrays = load_parameters(artifact)
    mlir = render_mlir(source, scope)

    output_mlir.parent.mkdir(parents=True, exist_ok=True)
    output_mlir.write_text(mlir)
    write_archive(output_irpa, arrays)
    return sum(int(array.nbytes) for array in arrays.values())


def main() -> int:
    args = parse_args()
    parameter_bytes = build_bundle(
        args.artifact, args.source, args.output_mlir, args.output_irpa, args.scope
    )
    print(f"wrote_mlir={args.output_mlir}")
    print(f"wrote_irpa={args.output_irpa}")
    print(f"scope={args.scope}")
    print(f"canonical_parameter_bytes={parameter_bytes}")
    print("duplicated_parameter_bytes=0")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
