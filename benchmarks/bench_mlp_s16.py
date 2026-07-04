#!/usr/bin/env python3
"""Run the first Stage A S=16 MLIR benchmark.

This intentionally measures only the current MLIR runner path. It does not make
backend claims beyond this lowering pipeline.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {
    "scalar": (
        "-one-shot-bufferize=bufferize-function-boundaries",
        "-buffer-deallocation-pipeline",
        "-convert-bufferization-to-memref",
        "-convert-linalg-to-loops",
        "-convert-scf-to-cf",
        "-expand-strided-metadata",
        "-lower-affine",
        "-convert-math-to-libm",
        "-convert-vector-to-llvm",
        "-convert-arith-to-llvm",
        "-finalize-memref-to-llvm",
        "-convert-func-to-llvm",
        "-convert-cf-to-llvm",
        "-reconcile-unrealized-casts",
    ),
    "affine": (
        "-one-shot-bufferize=bufferize-function-boundaries",
        "-buffer-deallocation-pipeline",
        "-convert-bufferization-to-memref",
        "-canonicalize",
        "-cse",
        "-convert-linalg-to-affine-loops",
        "-affine-simplify-structures",
        "-affine-scalrep",
        "-canonicalize",
        "-cse",
        "-convert-scf-to-cf",
        "-expand-strided-metadata",
        "-lower-affine",
        "-convert-scf-to-cf",
        "-convert-math-to-libm",
        "-convert-vector-to-llvm",
        "-convert-arith-to-llvm",
        "-finalize-memref-to-llvm",
        "-convert-func-to-llvm",
        "-convert-cf-to-llvm",
        "-reconcile-unrealized-casts",
    ),
}
PARAM_TYPES = (
    "tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>"
)
COUNT_PATTERNS = (
    ("tensor.dim", ("tensor.dim",)),
    ("memref.dim", ("memref.dim",)),
    ("branches", ("cf.cond_br", "llvm.cond_br")),
    ("calls", ("func.call", "llvm.call")),
    ("allocs", ("memref.alloc", "llvm.alloca", "llvm.call @malloc")),
    ("frees", ("memref.dealloc", "llvm.call @free")),
)


@dataclass(frozen=True)
class Variant:
    name: str
    input_type: str
    result_type: str
    entry: str
    source: Path
    run_specializer: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlir-opt", default=REPO_ROOT / "build/llvm/bin/mlir-opt", type=Path
    )
    parser.add_argument(
        "--mlir-runner", default=REPO_ROOT / "build/llvm/bin/mlir-runner", type=Path
    )
    parser.add_argument(
        "--plugin",
        default=REPO_ROOT / "build/shortseq-pinned/lib/ShortSeqPasses.so",
        type=Path,
    )
    parser.add_argument(
        "--runner-utils",
        default=REPO_ROOT / "build/llvm/lib/libmlir_runner_utils.so",
        type=Path,
    )
    parser.add_argument(
        "--c-runner-utils",
        default=REPO_ROOT / "build/llvm/lib/libmlir_c_runner_utils.so",
        type=Path,
    )
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="scalar")
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--iterations", default=25, type=int)
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument(
        "--cpu",
        type=int,
        help="CPU core for mlir-runner affinity; defaults to first available CPU",
    )
    parser.add_argument("--no-affinity", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument(
        "--dump-objects",
        action="store_true",
        help="Dump one JIT object per variant; requires --dump-dir",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def run_command(
    command: list[str],
    input_text: str | None = None,
    preexec_fn=None,
) -> str:
    completed = subprocess.run(
        command,
        input=input_text,
        check=False,
        preexec_fn=preexec_fn,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n" + " ".join(command) + "\n\nstderr:\n" + completed.stderr
        )
    return completed.stdout


def inject_main(module_ir: str, main_ir: str) -> str:
    text = module_ir.strip()
    module_start = text.find("module")
    if module_start == -1:
        return "module {\n" + text + "\n" + main_ir + "\n}\n"

    prefix = text[:module_start].strip()
    opener = re.search(
        r"\bmodule(?:\s+attributes\s+\{[^{}]*\})?\s*\{", text[module_start:]
    )
    if opener is None:
        raise ValueError("could not find MLIR module opener")
    start = module_start + opener.end() - 1
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("could not unwrap MLIR module")
    body = text[start + 1 : end].strip()

    prefix_text = f"{prefix}\n" if prefix else ""
    return prefix_text + "module {\n" + body + "\n" + main_ir + "\n}\n"


def specialize_dynamic(args: argparse.Namespace) -> str:
    return run_command(
        [
            str(args.mlir_opt),
            f"--load-pass-plugin={args.plugin}",
            "--pass-pipeline=builtin.module(shortseq-specialize)",
            str(REPO_ROOT / "examples/mlp/dynamic_mlp.mlir"),
        ]
    )


def main_function(variant: Variant, warmup: int, iterations: int) -> str:
    input_value = "%x_dyn" if "?" in variant.input_type else "%x"
    input_init = (
        "    %x_dyn = tensor.cast %x : tensor<1x16x384xf32> to tensor<1x?x384xf32>\n"
        if "?" in variant.input_type
        else ""
    )
    result_cast = (
        "      %y_static = tensor.cast %y : tensor<1x?x384xf32> to tensor<1x16x384xf32>\n"
        if "?" in variant.result_type
        else ""
    )
    extract_source = "%y_static" if "?" in variant.result_type else "%y"

    return f"""
  func.func @main() {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %warmup_n = arith.constant {warmup} : index
    %timed_n = arith.constant {iterations} : index

    %xv = arith.constant 1.000000e-01 : f32
    %w1v = arith.constant 2.000000e-02 : f32
    %b1v = arith.constant 1.000000e-02 : f32
    %w2v = arith.constant 1.500000e-02 : f32
    %b2v = arith.constant 5.000000e-03 : f32

    %x_empty = tensor.empty() : tensor<1x16x384xf32>
    %w1_empty = tensor.empty() : tensor<384x1536xf32>
    %b1_empty = tensor.empty() : tensor<1536xf32>
    %w2_empty = tensor.empty() : tensor<1536x384xf32>
    %b2_empty = tensor.empty() : tensor<384xf32>

    %x = linalg.fill ins(%xv : f32) outs(%x_empty : tensor<1x16x384xf32>) -> tensor<1x16x384xf32>
    %w1 = linalg.fill ins(%w1v : f32) outs(%w1_empty : tensor<384x1536xf32>) -> tensor<384x1536xf32>
    %b1 = linalg.fill ins(%b1v : f32) outs(%b1_empty : tensor<1536xf32>) -> tensor<1536xf32>
    %w2 = linalg.fill ins(%w2v : f32) outs(%w2_empty : tensor<1536x384xf32>) -> tensor<1536x384xf32>
    %b2 = linalg.fill ins(%b2v : f32) outs(%b2_empty : tensor<384xf32>) -> tensor<384xf32>
{input_init.rstrip()}

    %checksum0 = arith.constant 0.000000e+00 : f32
    %warmup_checksum = scf.for %i = %c0 to %warmup_n step %c1 iter_args(%acc = %checksum0) -> (f32) {{
      %y = func.call @{variant.entry}({input_value}, %w1, %b1, %w2, %b2)
          : ({variant.input_type}, {PARAM_TYPES}) -> {variant.result_type}
{result_cast.rstrip()}
      %result0 = tensor.extract {extract_source}[%c0, %c0, %c0] : tensor<1x16x384xf32>
      %next = arith.addf %acc, %result0 : f32
      scf.yield %next : f32
    }}

    %start = func.call @rtclock() : () -> f64
    %checksum = scf.for %i = %c0 to %timed_n step %c1 iter_args(%acc = %warmup_checksum) -> (f32) {{
      %y = func.call @{variant.entry}({input_value}, %w1, %b1, %w2, %b2)
          : ({variant.input_type}, {PARAM_TYPES}) -> {variant.result_type}
{result_cast.rstrip()}
      %result0 = tensor.extract {extract_source}[%c0, %c0, %c0] : tensor<1x16x384xf32>
      %next = arith.addf %acc, %result0 : f32
      scf.yield %next : f32
    }}
    %end = func.call @rtclock() : () -> f64
    %elapsed = arith.subf %end, %start : f64

    vector.print str "elapsed_seconds\\n"
    vector.print %elapsed : f64
    vector.print str "checksum\\n"
    vector.print %checksum : f32
    return
  }}

  func.func private @rtclock() -> f64
"""


def benchmark_module(
    variant: Variant, source_ir: str, warmup: int, iterations: int
) -> str:
    return inject_main(source_ir, main_function(variant, warmup, iterations))


def lower_module(args: argparse.Namespace, mlir: str) -> str:
    return run_command([str(args.mlir_opt), *PIPELINES[args.pipeline]], input_text=mlir)


def artifact_summary(module: str, lowered: str) -> str:
    counts = {
        label: sum(lowered.count(pattern) for pattern in patterns)
        for label, patterns in COUNT_PATTERNS
    }
    count_text = " ".join(f"{label}={counts[label]}" for label, _ in COUNT_PATTERNS)
    return (
        f"mlir_bytes={len(module.encode())} "
        f"lowered_mlir_bytes={len(lowered.encode())} "
        f"{count_text}"
    )


def available_cpus() -> set[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return set(os.sched_getaffinity(0))


def select_affinity_cpu(args: argparse.Namespace) -> int | None:
    if args.no_affinity:
        return None

    cpus = available_cpus()
    if cpus is None:
        if args.cpu is not None:
            raise RuntimeError("CPU affinity is not supported on this platform")
        return None

    cpu = min(cpus) if args.cpu is None else args.cpu
    if cpu not in cpus:
        available = ", ".join(str(value) for value in sorted(cpus))
        raise RuntimeError(
            f"CPU {cpu} is outside the current affinity set: {available}"
        )
    return cpu


def pin_to_cpu(cpu: int | None):
    if cpu is None:
        return None

    def preexec() -> None:
        os.sched_setaffinity(0, {cpu})

    return preexec


def run_lowered(
    args: argparse.Namespace, lowered: str, affinity_cpu: int | None
) -> tuple[float, float]:
    output = run_command(
        [
            str(args.mlir_runner),
            "-O3",
            "-e",
            "main",
            "-entry-point-result=void",
            f"-shared-libs={args.c_runner_utils},{args.runner_utils}",
        ],
        input_text=lowered,
        preexec_fn=pin_to_cpu(affinity_cpu),
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    elapsed = float(lines[lines.index("elapsed_seconds") + 1])
    checksum = float(lines[lines.index("checksum") + 1])
    return elapsed, checksum


def dump_object(args: argparse.Namespace, lowered: str, object_path: Path) -> int:
    run_command(
        [
            str(args.mlir_runner),
            "-O3",
            "-e",
            "main",
            "-entry-point-result=void",
            "-dump-object-file",
            f"-object-filename={object_path}",
            f"-shared-libs={args.c_runner_utils},{args.runner_utils}",
        ],
        input_text=lowered,
    )
    return object_path.stat().st_size


def temp_dir_context(dump_dir: Path | None):
    if dump_dir is None:
        return tempfile.TemporaryDirectory(prefix="shortseq-bench-")

    dump_dir.mkdir(parents=True, exist_ok=True)
    return nullcontext(str(dump_dir))


def summarize(samples: list[float], iterations: int) -> str:
    per_iter_ms = [sample * 1000.0 / iterations for sample in samples]
    return (
        f"median_ms={statistics.median(per_iter_ms):.6f} "
        f"min_ms={min(per_iter_ms):.6f} "
        f"max_ms={max(per_iter_ms):.6f}"
    )


def main() -> int:
    args = parse_args()
    if args.dump_objects and args.dump_dir is None:
        raise RuntimeError("--dump-objects requires --dump-dir")

    for path in (
        args.mlir_opt,
        args.mlir_runner,
        args.plugin,
        args.runner_utils,
        args.c_runner_utils,
    ):
        require_file(path)

    affinity_cpu = select_affinity_cpu(args)
    affinity_label = "off" if affinity_cpu is None else f"cpu{affinity_cpu}"

    variants = (
        Variant(
            "dynamic_generic",
            "tensor<1x?x384xf32>",
            "tensor<1x?x384xf32>",
            "mlp",
            REPO_ROOT / "examples/mlp/dynamic_mlp.mlir",
            False,
        ),
        Variant(
            "static_oracle",
            "tensor<1x16x384xf32>",
            "tensor<1x16x384xf32>",
            "mlp_s16",
            REPO_ROOT / "examples/mlp/static_mlp_s16.mlir",
            False,
        ),
        Variant(
            "dispatched_wrapper",
            "tensor<1x?x384xf32>",
            "tensor<1x?x384xf32>",
            "mlp",
            REPO_ROOT / "examples/mlp/dynamic_mlp.mlir",
            True,
        ),
    )

    with temp_dir_context(args.dump_dir) as tmp:
        tmpdir = Path(tmp)
        if args.dump_dir is not None:
            print(f"dump directory: {tmpdir}")

        print(
            f"S=16 benchmark, pipeline={args.pipeline}, warmup={args.warmup}, "
            f"iterations={args.iterations}, repeats={args.repeats}, "
            f"affinity={affinity_label}"
        )
        for variant in variants:
            if variant.run_specializer:
                source_ir = specialize_dynamic(args)
            else:
                source_ir = variant.source.read_text()
            module = benchmark_module(variant, source_ir, args.warmup, args.iterations)
            lowered = lower_module(args, module)
            object_size = None
            if args.dump_dir is not None:
                (tmpdir / f"{variant.name}.mlir").write_text(module)
                (tmpdir / f"{variant.name}.llvm.mlir").write_text(lowered)
            if args.dump_objects:
                object_size = dump_object(args, lowered, tmpdir / f"{variant.name}.o")

            samples: list[float] = []
            checksum: float | None = None
            for _ in range(args.repeats):
                elapsed, current_checksum = run_lowered(args, lowered, affinity_cpu)
                samples.append(elapsed)
                checksum = current_checksum

            object_text = ""
            if object_size is not None:
                object_text = f" object_bytes={object_size}"
            print(
                f"{variant.name}: {summarize(samples, args.iterations)} "
                f"checksum={checksum:.8g} {artifact_summary(module, lowered)}"
                f"{object_text}"
            )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
