#!/usr/bin/env python3
"""Benchmark Stage A MLP variants with IREE llvm-cpu."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAM_INPUTS = (
    "--input=384x1536xf32",
    "--input=1536xf32",
    "--input=1536x384xf32",
    "--input=384xf32",
)
TIME_SCALE_TO_MS = {
    "ns": 1.0e-6,
    "us": 1.0e-3,
    "ms": 1.0,
    "s": 1.0e3,
}


@dataclass(frozen=True)
class Variant:
    name: str
    entry: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="16", help="Comma-separated sequence lengths")
    parser.add_argument("--mlir-opt", default=REPO_ROOT / "build/llvm/bin/mlir-opt", type=Path)
    parser.add_argument(
        "--plugin",
        default=REPO_ROOT / "build/shortseq-pinned/lib/ShortSeqPasses.so",
        type=Path,
    )
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--iree-benchmark", default="iree-benchmark-module")
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--no-affinity", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    return parser.parse_args()


def parse_lengths(text: str) -> list[int]:
    lengths = []
    for item in text.split(","):
        value = int(item.strip())
        if value <= 0:
            raise ValueError("--lengths values must be positive")
        lengths.append(value)
    return sorted(set(lengths))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        raise FileNotFoundError(tool)


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


def specialize_s16(args: argparse.Namespace) -> str:
    return run_command(
        [
            str(args.mlir_opt),
            f"--load-pass-plugin={args.plugin}",
            "--pass-pipeline=builtin.module(shortseq-specialize)",
            str(REPO_ROOT / "examples/mlp/dynamic_mlp.mlir"),
        ]
    )


def static_oracle_source(sequence_length: int) -> str:
    source = (REPO_ROOT / "examples/mlp/static_mlp_s16.mlir").read_text()
    if sequence_length == 16:
        return source
    replacements = (
        ("mlp_s16", f"mlp_s{sequence_length}"),
        ("1x16x384", f"1x{sequence_length}x384"),
        ("16x1536", f"{sequence_length}x1536"),
        ("16x384", f"{sequence_length}x384"),
        ("[1, 16, 384]", f"[1, {sequence_length}, 384]"),
        ("[16, 1536]", f"[{sequence_length}, 1536]"),
        ("[16, 384]", f"[{sequence_length}, 384]"),
    )
    for old, new in replacements:
        source = source.replace(old, new)
    return source


def variants_for(args: argparse.Namespace, sequence_length: int) -> tuple[Variant, ...]:
    dynamic = (REPO_ROOT / "examples/mlp/dynamic_mlp.mlir").read_text()
    variants = [
        Variant("dynamic_generic", "mlp", dynamic),
        Variant("static_oracle", f"mlp_s{sequence_length}", static_oracle_source(sequence_length)),
    ]
    if sequence_length == 16:
        variants.append(Variant("dispatched_wrapper", "mlp", specialize_s16(args)))
    return tuple(variants)


def compile_iree(args: argparse.Namespace, source: str, output_path: Path) -> float:
    start = time.perf_counter()
    run_command(
        [
            args.iree_compile,
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
            "-o",
            str(output_path),
            "-",
        ],
        input_text=source,
    )
    return time.perf_counter() - start


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
        raise RuntimeError(f"CPU {cpu} is outside the current affinity set: {available}")
    return cpu


def pin_to_cpu(cpu: int | None):
    if cpu is None:
        return None

    def preexec() -> None:
        os.sched_setaffinity(0, {cpu})

    return preexec


def benchmark_ms(
    args: argparse.Namespace,
    vmfb_path: Path,
    entry: str,
    sequence_length: int,
    affinity_cpu: int | None,
) -> float:
    output = run_command(
        [
            args.iree_benchmark,
            f"--module={vmfb_path}",
            f"--function={entry}",
            f"--input=1x{sequence_length}x384xf32",
            *PARAM_INPUTS,
            "--benchmark_format=json",
        ],
        preexec_fn=pin_to_cpu(affinity_cpu),
    )
    json_start = output.find("{")
    if json_start == -1:
        raise RuntimeError("iree-benchmark-module did not emit JSON")
    record = json.loads(output[json_start:])["benchmarks"][0]
    return float(record["real_time"]) * TIME_SCALE_TO_MS[record["time_unit"]]


def temp_dir_context(dump_dir: Path | None):
    if dump_dir is None:
        return tempfile.TemporaryDirectory(prefix="iree-bench-")
    dump_dir.mkdir(parents=True, exist_ok=True)
    return nullcontext(str(dump_dir))


def summarize(samples: list[float]) -> str:
    return (
        f"median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} "
        f"max_ms={max(samples):.6f}"
    )


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.lengths)

    require_tool(args.iree_compile)
    require_tool(args.iree_benchmark)
    require_file(args.mlir_opt)
    require_file(args.plugin)

    affinity_cpu = select_affinity_cpu(args)
    affinity_label = "off" if affinity_cpu is None else f"cpu{affinity_cpu}"

    with temp_dir_context(args.dump_dir) as tmp:
        tmpdir = Path(tmp)
        if args.dump_dir is not None:
            print(f"dump directory: {tmpdir}")

        print(
            f"IREE benchmark, lengths={','.join(str(v) for v in lengths)}, "
            f"repeats={args.repeats}, affinity={affinity_label}"
        )
        for sequence_length in lengths:
            print(f"S={sequence_length}")
            for variant in variants_for(args, sequence_length):
                prefix = tmpdir / f"s{sequence_length}_{variant.name}"
                if args.dump_dir is not None:
                    prefix.with_suffix(".mlir").write_text(variant.source)

                vmfb_path = prefix.with_suffix(".vmfb")
                compile_s = compile_iree(args, variant.source, vmfb_path)
                samples = [
                    benchmark_ms(args, vmfb_path, variant.entry, sequence_length, affinity_cpu)
                    for _ in range(args.repeats)
                ]
                print(
                    f"{variant.name}: {summarize(samples)} "
                    f"compile_s={compile_s:.3f} vmfb_bytes={vmfb_path.stat().st_size}"
                )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
