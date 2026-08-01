#!/usr/bin/env python3
"""Build independent fixed-width E5 core MLIR files for benchmarking."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import onnx

from adapt_e5_onnx import assert_initializer_identity, validate_dynamic_contract
from build_e5_bridge import TORCH_TO_BUILTIN_PIPELINE, require_tool, run
from e5_common import DEFAULT_ARTIFACT_DIR, REPO_ROOT, sha256_file

REFERENCE_RE = re.compile(r'#stream\.parameter\.named<"e5"::"([^"]+)">')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--lengths", default="16,32,64,128")
    parser.add_argument(
        "--mlir-opt", type=Path, default=REPO_ROOT / "build/llvm/bin/mlir-opt"
    )
    return parser.parse_args()


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item) for item in text.split(",")]
    if not lengths or any(length <= 0 or length > 512 for length in lengths):
        raise ValueError("--lengths must contain values in [1,512]")
    if len(lengths) != len(set(lengths)):
        raise ValueError("--lengths contains duplicates")
    return sorted(lengths)


def staticize_model(source: Path, output: Path, length: int) -> set[str]:
    model = onnx.load(source, load_external_data=False)
    validate_dynamic_contract(model)
    initializers = {
        item.name: item.SerializeToString() for item in model.graph.initializer
    }

    changed = 0
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    for value in values:
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.HasField("dim_param"):
                dimension.dim_value = length
                changed += 1
    if changed == 0:
        raise AssertionError("adapted E5 ONNX has no symbolic sequence dimensions")

    onnx.checker.check_model(model, full_check=True)
    assert_initializer_identity(initializers, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return {name.replace("::", "__") for name in initializers}


def check_core(core_path: Path, length: int, initializer_keys: set[str]) -> None:
    core = core_path.read_text(encoding="utf-8")
    signature = (
        "func.func @sentence_embedding("
        f"%arg0: tensor<1x{length}xi64>, %arg1: tensor<1x{length}xi64>, "
        f"%arg2: tensor<1x{length}xi64>) -> tensor<1x384xf32>"
    )
    if core.count(signature) != 1:
        raise AssertionError(f"S={length} oracle has the wrong entry signature")
    if "?" in core:
        raise AssertionError(f"S={length} oracle retains a dynamic dimension")
    references = REFERENCE_RE.findall(core)
    if set(references) != initializer_keys or len(references) != len(initializer_keys):
        raise AssertionError(f"S={length} oracle has the wrong parameter references")
    if core.count('"util.global"') != len(initializer_keys):
        raise AssertionError(f"S={length} oracle must declare each parameter once")
    if "dense_resource<" in core:
        raise AssertionError(f"S={length} oracle contains an embedded dense resource")


def build_oracle(
    artifact_dir: Path,
    length: int,
    mlir_opt: Path,
    importer: str,
    optimizer: str,
    log: Path,
) -> dict[str, object]:
    static_onnx = artifact_dir / f"sentence_embedding.static_s{length}.onnx"
    imported = artifact_dir / f"sentence_embedding.static_s{length}.imported.mlir"
    torch_backend = artifact_dir / f"sentence_embedding.static_s{length}.torch.mlir"
    builtin_core = artifact_dir / f"sentence_embedding.static_s{length}.iree.mlir"
    generic_core = artifact_dir / f"sentence_embedding.static_s{length}.generic.mlir"
    core = artifact_dir / f"sentence_embedding.static_s{length}.core.mlir"
    existing_irpas = set(artifact_dir.glob("*.irpa"))
    if existing_irpas != {artifact_dir / "e5.irpa"}:
        raise AssertionError("static oracles require exactly one canonical e5.irpa")
    canonical_irpa_hash = sha256_file(artifact_dir / "e5.irpa")

    start = time.perf_counter()
    initializer_keys = staticize_model(
        artifact_dir / "sentence_embedding.onnx", static_onnx, length
    )
    run(
        [
            importer,
            str(static_onnx),
            "--opset-version",
            "17",
            "--externalize-params",
            "--params-scope",
            "e5",
            "--no-save-params",
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
    run(
        [
            optimizer,
            str(builtin_core),
            "--mlir-print-op-generic",
            "-o",
            str(generic_core),
        ],
        log,
    )
    run(
        [
            str(mlir_opt),
            "--allow-unregistered-dialect",
            str(generic_core),
            "-o",
            str(core),
        ],
        log,
    )
    check_core(core, length, initializer_keys)
    if set(artifact_dir.glob("*.irpa")) != existing_irpas:
        raise AssertionError("static oracle import created a second IRPA archive")
    if sha256_file(artifact_dir / "e5.irpa") != canonical_irpa_hash:
        raise AssertionError("static oracle import changed canonical e5.irpa bytes")
    return {
        "length": length,
        "frontend_s": time.perf_counter() - start,
        "core": str(core),
        "initializer_count": len(initializer_keys),
        "new_parameter_archives": 0,
        "canonical_irpa_sha256": canonical_irpa_hash,
    }


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    importer = require_tool("iree-import-onnx")
    optimizer = require_tool("iree-opt")
    if not args.mlir_opt.is_file():
        raise AssertionError(f"upstream mlir-opt not found: {args.mlir_opt}")
    if not (args.artifact_dir / "e5.irpa").is_file():
        raise AssertionError("missing canonical e5.irpa; run check-e5 first")

    log = args.artifact_dir / "static_oracles.log"
    log.unlink(missing_ok=True)
    records = [
        build_oracle(
            args.artifact_dir,
            length,
            args.mlir_opt,
            importer,
            optimizer,
            log,
        )
        for length in lengths
    ]
    manifest = args.artifact_dir / "static_oracles.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(
            f"PASS static oracle S={record['length']} "
            f"frontend_s={record['frontend_s']:.3f}"
        )
    print("shared_parameter_archive=e5.irpa")
    print("new_parameter_archives=0")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"error: oracle command exited {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
