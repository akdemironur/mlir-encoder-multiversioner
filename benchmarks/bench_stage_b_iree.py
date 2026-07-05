#!/usr/bin/env python3
"""Benchmark the Stage B contract fixture with IREE."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import iree.compiler as ireec
    import iree.runtime as ireert
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "IREE benchmark dependencies are required. Run with: "
        "uv run --frozen --group bench python benchmarks/bench_stage_b_iree.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "examples/stage_b/contract_encoder_block.mlir"
IREE_FLAGS = (
    "--iree-llvmcpu-target-cpu=host",
    "--iree-scheduling-optimize-bindings=false",
)
PARAMS = (
    ("q_w", (64, 64)),
    ("k_w", (64, 64)),
    ("v_w", (64, 64)),
    ("o_w", (64, 64)),
    ("norm_scale", (64,)),
    ("norm_bias", (64,)),
    ("ff_w1", (64, 256)),
    ("ff_b1", (256,)),
    ("ff_w2", (256, 64)),
    ("ff_b2", (64,)),
)


@dataclass(frozen=True)
class CompiledModule:
    name: str
    function: object
    compile_s: float
    vmfb_bytes: int
    keepalive: tuple[object, object, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default=REPO_ROOT / "results/stage_b/encoder_block_synthetic.npz",
        type=Path,
    )
    parser.add_argument("--lengths", default="4,8")
    parser.add_argument("--mlir-opt", default=REPO_ROOT / "build/llvm/bin/mlir-opt", type=Path)
    parser.add_argument(
        "--plugin",
        default=REPO_ROOT / "build/shortseq-pinned/lib/ShortSeqPasses.so",
        type=Path,
    )
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--iterations", default=50, type=int)
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--no-affinity", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    return parser.parse_args()


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item) for item in text.split(",") if item]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("--lengths contains duplicates")
    return sorted(lengths)


def check_args(args: argparse.Namespace) -> None:
    if not args.artifact.exists():
        raise FileNotFoundError(
            f"missing artifact: {args.artifact}\n"
            "generate it with:\n"
            "  uv run --frozen --group numerical python "
            "scripts/make_stage_b_artifact.py "
            "--output results/stage_b/encoder_block_synthetic.npz "
            f"--lengths {args.lengths}\n"
            "then check it with:\n"
            "  uv run --frozen --group numerical python "
            "scripts/check_stage_b_artifact.py "
            "--input results/stage_b/encoder_block_synthetic.npz"
        )

    for path in (args.mlir_opt, args.plugin, SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("--warmup must be >= 0; --iterations and --repeats must be > 0")


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n" + " ".join(command) + "\n\nstderr:\n" + completed.stderr
        )
    return completed.stdout


def stage_b_sources(args: argparse.Namespace, lengths: list[int]) -> dict[str, str]:
    pipeline = "builtin.module(shortseq-specialize{lengths=" + ",".join(map(str, lengths)) + "})"
    wrapper = run_command(
        [
            str(args.mlir_opt),
            f"--load-pass-plugin={args.plugin}",
            f"--pass-pipeline={pipeline}",
            str(SOURCE),
        ]
    )
    return {
        "dynamic_generic": SOURCE.read_text(),
        "dispatched_wrapper": wrapper,
    }


def require_array(
    data: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    if name not in data.files:
        raise AssertionError(f"missing {name}")
    array = data[name]
    if array.shape != shape:
        raise AssertionError(f"{name} shape {array.shape}, expected {shape}")
    if array.dtype != np.float32:
        raise AssertionError(f"{name} dtype {array.dtype}, expected float32")
    return array


def load_artifact(path: Path, lengths: list[int]):
    with np.load(path) as data:
        params = tuple(require_array(data, name, shape) for name, shape in PARAMS)
        inputs = {
            length: require_array(data, f"x_s{length}", (1, length, 64))
            for length in lengths
        }
        copied_params = [
            key
            for key in data.files
            if re.fullmatch(r".*_s[1-9][0-9]*", key) and not key.startswith("x_s")
        ]
        if copied_params:
            raise AssertionError(f"length-specific parameter keys: {copied_params}")

    parameter_bytes = sum(array.nbytes for array in params)
    sample_input_bytes = sum(array.nbytes for array in inputs.values())
    return params, inputs, parameter_bytes, sample_input_bytes


def set_affinity(args: argparse.Namespace) -> int | None:
    if args.no_affinity:
        return None
    if not hasattr(os, "sched_getaffinity"):
        if args.cpu is not None:
            raise RuntimeError("CPU affinity is not supported on this platform")
        return None

    allowed = set(os.sched_getaffinity(0))
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        available = ",".join(str(value) for value in sorted(allowed))
        raise RuntimeError(f"CPU {cpu} is outside the current affinity set: {available}")
    os.sched_setaffinity(0, {cpu})
    return cpu


def compile_source(name: str, source: str, dump_dir: Path) -> CompiledModule:
    start = time.perf_counter()
    vmfb = bytes(
        ireec.compile_str(
            source,
            target_backends=["llvm-cpu"],
            extra_args=list(IREE_FLAGS),
        )
    )
    compile_s = time.perf_counter() - start

    (dump_dir / f"{name}.mlir").write_text(source)
    (dump_dir / f"{name}.vmfb").write_bytes(vmfb)

    config = ireert.Config("local-sync")
    module = ireert.VmModule.copy_buffer(config.vm_instance, vmfb)
    context = ireert.SystemContext(config=config)
    context.add_vm_module(module)
    return CompiledModule(
        name=name,
        function=context.modules.module.encoder_block,
        compile_s=compile_s,
        vmfb_bytes=len(vmfb),
        keepalive=(config, module, context),
    )


def benchmark_ms(function, x, params, warmup: int, iterations: int, repeats: int):
    result = None
    for _ in range(warmup):
        result = function(x, *params)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            result = function(x, *params)
        samples.append((time.perf_counter_ns() - start) / iterations / 1.0e6)

    return samples, float(np.asarray(result).sum())


def sample_stats(samples: list[float]) -> tuple[float, float, float]:
    return statistics.median(samples), min(samples), max(samples)


def error_line(exc: Exception) -> str:
    for line in str(exc).splitlines():
        if line.strip():
            return line.strip().split("; while ", maxsplit=1)[0]
    return type(exc).__name__


def print_compile_summary(dynamic: CompiledModule, wrapper: CompiledModule) -> None:
    print("compiled modules:")
    print("variant              compile_s  vmfb_bytes")
    for variant in (dynamic, wrapper):
        print(
            f"{variant.name:<20}"
            f"{variant.compile_s:>9.3f}"
            f"{variant.vmfb_bytes:>12}"
        )
    print()


def print_latency_header() -> None:
    print("latency_ms:")
    print(
        "S      dyn_med  dyn_min  dyn_max  "
        "wrap_med  wrap_min  wrap_max  speedup  checksum"
    )


def print_latency_row(
    length: int,
    dynamic_samples: list[float],
    wrapper_samples: list[float],
    checksum: float,
) -> None:
    dyn_med, dyn_min, dyn_max = sample_stats(dynamic_samples)
    wrap_med, wrap_min, wrap_max = sample_stats(wrapper_samples)
    print(
        f"{length:<5}"
        f"{dyn_med:>9.6f}{dyn_min:>9.6f}{dyn_max:>9.6f}"
        f"{wrap_med:>10.6f}{wrap_min:>10.6f}{wrap_max:>10.6f}"
        f"{(dyn_med / wrap_med):>8.3f}x"
        f"{checksum:>11.6f}"
    )


def print_latency_error(length: int, variant: CompiledModule, exc: Exception) -> None:
    print(f"{length:<5}{variant.name}: ERROR {error_line(exc)}")


def dump_context(path: Path | None):
    if path is None:
        return TemporaryDirectory(prefix="stage-b-iree-")
    path.mkdir(parents=True, exist_ok=True)
    return nullcontext(str(path))


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    check_args(args)

    params, inputs, parameter_bytes, sample_input_bytes = load_artifact(
        args.artifact, lengths
    )
    affinity_cpu = set_affinity(args)
    affinity = "off" if affinity_cpu is None else f"cpu{affinity_cpu}"

    with dump_context(args.dump_dir) as tmp:
        dump_dir = Path(tmp)
        sources = stage_b_sources(args, lengths)
        dynamic = compile_source("dynamic_generic", sources["dynamic_generic"], dump_dir)
        wrapper = compile_source(
            "dispatched_wrapper", sources["dispatched_wrapper"], dump_dir
        )
        if args.dump_dir is not None:
            print(f"dump directory: {dump_dir}")

        print(
            f"Stage B IREE benchmark, lengths={','.join(map(str, lengths))}, "
            f"warmup={args.warmup}, iterations={args.iterations}, "
            f"repeats={args.repeats}, affinity={affinity}"
        )
        print(
            "modules=2 compiled_once=true "
            f"dispatched_wrapper_branches={','.join(map(str, lengths))}"
        )
        print(f"iree_flags={' '.join(IREE_FLAGS)}")
        print(
            f"artifact={args.artifact} parameter_bytes={parameter_bytes} "
            f"sample_input_bytes={sample_input_bytes}"
        )
        print()
        print_compile_summary(dynamic, wrapper)

        failed = False
        print_latency_header()
        for length in lengths:
            x = inputs[length]
            try:
                reference = np.asarray(dynamic.function(x, *params))
                dynamic_samples, checksum = benchmark_ms(
                    dynamic.function,
                    x,
                    params,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )

                output = np.asarray(wrapper.function(x, *params))
                np.testing.assert_allclose(output, reference, rtol=1.0e-4, atol=1.0e-4)
                wrapper_samples, wrapper_checksum = benchmark_ms(
                    wrapper.function,
                    x,
                    params,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                np.testing.assert_allclose(
                    wrapper_checksum, checksum, rtol=1.0e-4, atol=1.0e-4
                )
                print_latency_row(length, dynamic_samples, wrapper_samples, checksum)
            except Exception as exc:
                failed = True
                print_latency_error(length, wrapper, exc)

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
