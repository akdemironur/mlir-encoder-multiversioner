#!/usr/bin/env python3
"""Build the fixed E5 IREE core boundary, IRPA, and dynamic baseline."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from e5_common import DEFAULT_ARTIFACT_DIR, REPO_ROOT


# Stop Torch lowering before IREE converts func.func into its HAL ABI. This
# short, model-audited suffix is enough after the standard ONNX-to-Torch
# backend pipeline and leaves a plain builtin-tensor func.func for our pass.
TORCH_TO_BUILTIN_PIPELINE = (
    "builtin.module("
    "func.func("
    "torch-scalarize-shapes,"
    "convert-torch-to-tmtensor,"
    "torch-iree-tm-tensor-to-linalg-ext,"
    "convert-torch-to-tensor,"
    "torch-iree-torch-unstructured-to-linalg-ext,"
    "convert-torch-to-linalg,"
    "convert-torch-to-scf,"
    "convert-torch-to-arith),"
    "convert-torch-conversion-to-mlprogram,"
    "func.func(canonicalize,resolve-shaped-type-result-dims,cse),"
    "inline,"
    "torch-func-backend-type-conversion,"
    "func.func(torch-finalizing-backend-type-conversion),"
    "canonicalize)"
)

ENTRY_SIGNATURE = (
    "func.func @sentence_embedding("
    "%arg0: tensor<1x?xi64>, %arg1: tensor<1x?xi64>, "
    "%arg2: tensor<1x?xi64>) -> tensor<1x384xf32> {"
)
MARKED_ENTRY_SIGNATURE = ENTRY_SIGNATURE[:-1] + (
    "attributes {shortseq.e5_small_v2, shortseq.entry} {"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--mlir-opt", type=Path, default=REPO_ROOT / "build/llvm/bin/mlir-opt"
    )
    return parser.parse_args()


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AssertionError(f"required tool not found: {name}")
    return path


def run(command: list[str], log: Path) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    with log.open("a", encoding="utf-8") as stream:
        stream.write("+ " + " ".join(command) + "\n")
        stream.write(completed.stdout)
    if completed.stdout:
        print(completed.stdout, end="")
    completed.check_returncode()


def mark_entry(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if text.count(ENTRY_SIGNATURE) != 1:
        raise AssertionError("expected exactly one canonical sentence_embedding entry")
    output.write_text(
        text.replace(ENTRY_SIGNATURE, MARKED_ENTRY_SIGNATURE), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    importer = require_tool("iree-import-onnx")
    optimizer = require_tool("iree-opt")
    compiler = require_tool("iree-compile")
    if not args.mlir_opt.is_file():
        raise AssertionError(f"upstream mlir-opt not found: {args.mlir_opt}")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    adapted = args.artifact_dir / "sentence_embedding.onnx"
    imported = args.artifact_dir / "sentence_embedding.imported.mlir"
    torch_backend = args.artifact_dir / "sentence_embedding.torch.mlir"
    builtin_core = args.artifact_dir / "sentence_embedding.core.iree.mlir"
    marked_core = args.artifact_dir / "sentence_embedding.marked.mlir"
    generic_core = args.artifact_dir / "sentence_embedding.generic.mlir"
    core = args.artifact_dir / "sentence_embedding.core.mlir"
    irpa = args.artifact_dir / "e5.irpa"
    vmfb = args.artifact_dir / "sentence_embedding.dynamic.vmfb"
    log = args.artifact_dir / "bridge.log"
    log.unlink(missing_ok=True)

    run(
        [
            importer,
            str(adapted),
            "--opset-version",
            "17",
            "--externalize-params",
            "--params-scope",
            "e5",
            "--save-params-to",
            str(irpa),
            "-o",
            str(imported),
        ],
        log,
    )
    run(
        [
            optimizer,
            str(imported),
            "--pass-pipeline=builtin.module(torch-onnx-to-torch-backend-pipeline)",
            "-o",
            str(torch_backend),
        ],
        log,
    )
    run(
        [
            optimizer,
            str(torch_backend),
            f"--pass-pipeline={TORCH_TO_BUILTIN_PIPELINE}",
            "-o",
            str(builtin_core),
        ],
        log,
    )
    mark_entry(builtin_core, marked_core)

    # IREE dialect operations use custom assembly. Print those operations in
    # generic form so upstream mlir-opt can preserve them while verifying the
    # builtin func/tensor/linalg IR that the specialization pass will edit.
    run(
        [
            optimizer,
            str(marked_core),
            "--mlir-print-op-generic",
            "-o",
            str(generic_core),
        ],
        log,
    )
    run(
        [
            str(args.mlir_opt),
            "--allow-unregistered-dialect",
            str(generic_core),
            "-o",
            str(core),
        ],
        log,
    )
    run(
        [
            compiler,
            str(core),
            "--iree-input-type=none",
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
            "--iree-scheduling-optimize-bindings=false",
            "-o",
            str(vmfb),
        ],
        log,
    )
    print(f"PASS E5 textual bridge: {core}")
    print(f"external_parameters={irpa}")
    print(f"dynamic_baseline={vmfb}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"error: bridge command exited {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
