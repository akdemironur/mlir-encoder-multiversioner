#!/usr/bin/env python3
"""Benchmark pinned E5 dynamic, static-oracle, and dispatched CPU modules."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from e5_common import DEFAULT_ARTIFACT_DIR  # noqa: E402
from validate_e5_inputs import validate_inputs  # noqa: E402

IREE_FLAGS = (
    "--iree-input-type=none",
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu=host",
    "--iree-scheduling-optimize-bindings=false",
)
VARIANT_ORDER = ("dynamic", "static-oracle", "dispatched")
EXACT_LENGTHS = (16, 32, 64, 128)
TIME_TO_MS = {"ns": 1.0e-6, "us": 1.0e-3, "ms": 1.0, "s": 1.0e3}


@dataclass
class CompiledArtifact:
    name: str
    variant: str
    source: Path
    vmfb: Path
    compile_s: float
    vmfb_bytes: int
    vm_flatbuffer_bytes: int
    vm_bytecode_bytes: int
    embedded_executable_bytes: int
    oracle_frontend_s: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--lengths", default="16,32,64,128")
    parser.add_argument(
        "--variants", default="dynamic,static-oracle,dispatched"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--benchmark-min-time", default="0.2s")
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--memory-warmup", type=int, default=2)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--no-affinity", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mlir-opt", type=Path, default=REPO_ROOT / "build/llvm/bin/mlir-opt"
    )
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--iree-benchmark", default="iree-benchmark-module")
    parser.add_argument("--iree-run", default="iree-run-module")
    parser.add_argument("--iree-dump-module", default="iree-dump-module")
    parser.add_argument("--iree-dump-parameters", default="iree-dump-parameters")

    # Internal mode used in a short-lived process to read RSS after warmup.
    parser.add_argument("--memory-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parameters", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ids", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mask", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--types", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item) for item in text.split(",")]
    if not lengths or any(length <= 0 or length > 512 for length in lengths):
        raise ValueError("--lengths must contain values in [1,512]")
    if len(lengths) != len(set(lengths)):
        raise ValueError("--lengths contains duplicates")
    return sorted(lengths)


def parse_variants(text: str) -> list[str]:
    requested = text.split(",")
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--variants must contain unique variant names")
    unknown = set(requested) - set(VARIANT_ORDER)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    return [variant for variant in VARIANT_ORDER if variant in requested]


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"required tool not found: {name}")
    return path


def select_cpu(args: argparse.Namespace) -> int | None:
    if args.no_affinity:
        return None
    if not hasattr(os, "sched_getaffinity"):
        if args.cpu is not None:
            raise RuntimeError("CPU affinity is unavailable on this platform")
        return None
    allowed = set(os.sched_getaffinity(0))
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        raise ValueError(f"CPU {cpu} is outside affinity set {sorted(allowed)}")
    return cpu


def pin_child(cpu: int | None):
    if cpu is None:
        return None

    def set_affinity() -> None:
        os.sched_setaffinity(0, {cpu})

    return set_affinity


def run_command(
    command: list[str],
    cpu: int | None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=pin_child(cpu),
    )
    if stdout_path is not None:
        stdout_path.write_text(completed.stdout, encoding="utf-8")
    if stderr_path is not None:
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\n\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return completed


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = args.artifact_dir / "benchmarks" / timestamp
    else:
        path = args.output_dir
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty benchmark dir: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_static_oracles(
    args: argparse.Namespace, lengths: list[int], output_dir: Path, cpu: int | None
) -> dict[int, float]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_e5_static_oracles.py"),
        "--artifact-dir",
        str(args.artifact_dir),
        "--lengths",
        ",".join(map(str, lengths)),
        "--mlir-opt",
        str(args.mlir_opt),
    ]
    run_command(
        command,
        cpu,
        output_dir / "static-oracle-build.stdout",
        output_dir / "static-oracle-build.stderr",
    )
    manifest = json.loads(
        (args.artifact_dir / "static_oracles.json").read_text(encoding="utf-8")
    )
    shutil.copy2(
        args.artifact_dir / "static_oracles.json",
        output_dir / "static-oracles.json",
    )
    return {int(item["length"]): float(item["frontend_s"]) for item in manifest}


def parse_module_metadata(text: str) -> tuple[int, int, int]:
    def value(pattern: str, name: str) -> int:
        match = re.search(pattern, text)
        if match is None:
            raise AssertionError(f"iree-dump-module omitted {name}")
        return int(match.group(1))

    return (
        value(r"FlatBuffer:\s+(\d+) bytes", "FlatBuffer bytes"),
        value(r"Bytecode:\s+(\d+) bytes", "bytecode bytes"),
        value(r"External \.rodata:\s+~?(\d+) bytes", "external rodata bytes"),
    )


def compile_artifact(
    name: str,
    variant: str,
    source: Path,
    output_dir: Path,
    compiler: str,
    dumper: str,
    cpu: int | None,
    oracle_frontend_s: float | None = None,
) -> CompiledArtifact:
    retained_source = output_dir / f"{name}.mlir"
    shutil.copy2(source, retained_source)
    vmfb = output_dir / f"{name}.vmfb"
    command = [compiler, str(retained_source), *IREE_FLAGS, "-o", str(vmfb)]
    start = time.perf_counter()
    run_command(
        command,
        cpu,
        output_dir / f"{name}.compile.stdout",
        output_dir / f"{name}.compile.stderr",
    )
    compile_s = time.perf_counter() - start
    metadata = run_command(
        [dumper, "--output=metadata", str(vmfb)], cpu
    ).stdout
    (output_dir / f"{name}.module-metadata.txt").write_text(
        metadata, encoding="utf-8"
    )
    flatbuffer, bytecode, executable = parse_module_metadata(metadata)
    return CompiledArtifact(
        name=name,
        variant=variant,
        source=retained_source,
        vmfb=vmfb,
        compile_s=compile_s,
        vmfb_bytes=vmfb.stat().st_size,
        vm_flatbuffer_bytes=flatbuffer,
        vm_bytecode_bytes=bytecode,
        embedded_executable_bytes=executable,
        oracle_frontend_s=oracle_frontend_s,
    )


def parameter_accounting(dumper: str, irpa: Path, cpu: int | None) -> dict[str, int]:
    output = run_command(
        [dumper, f"--parameters=e5={irpa}"], cpu
    ).stdout
    match = re.search(r"Parameter scope `e5` \((\d+) entries, (\d+) total bytes\)", output)
    if match is None:
        raise AssertionError("iree-dump-parameters omitted E5 archive totals")
    return {"irpa_entries": int(match.group(1)), "canonical_parameter_bytes": int(match.group(2))}


def validate_bridge_artifacts(
    args: argparse.Namespace,
    variants: list[str],
    output_dir: Path,
    cpu: int | None,
) -> None:
    base = [
        sys.executable,
        str(REPO_ROOT / "scripts/check_e5_bridge.py"),
        "--artifact-dir",
        str(args.artifact_dir),
    ]
    run_command(
        base,
        cpu,
        output_dir / "dynamic-artifact-check.stdout",
        output_dir / "dynamic-artifact-check.stderr",
    )
    if "dispatched" in variants:
        run_command(
            [
                *base,
                "--core",
                str(args.artifact_dir / "sentence_embedding.multiversioned.mlir"),
                "--specialized-lengths",
                ",".join(map(str, EXACT_LENGTHS)),
            ],
            cpu,
            output_dir / "dispatched-artifact-check.stdout",
            output_dir / "dispatched-artifact-check.stderr",
        )


def make_inputs(
    artifact_dir: Path, output_dir: Path, lengths: list[int]
) -> dict[int, dict[str, Path]]:
    session = ort.InferenceSession(
        str(artifact_dir / "sentence_embedding.onnx"),
        providers=["CPUExecutionProvider"],
    )
    fixtures: dict[int, dict[str, Path]] = {}
    for length in lengths:
        ids = ((np.arange(length, dtype=np.int64) * 37 + 101) % 30522)[None, :]
        mask = np.ones((1, length), dtype=np.int64)
        types = np.zeros((1, length), dtype=np.int64)
        validate_inputs(ids, mask, types)
        expected = session.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
        )[0]
        paths = {}
        for name, array in {
            "ids": ids,
            "mask": mask,
            "types": types,
            "expected": expected,
        }.items():
            path = output_dir / f"s{length}-{name}.npy"
            np.save(path, array)
            paths[name] = path
        fixtures[length] = paths
    return fixtures


def runtime_arguments(
    module: Path, irpa: Path, fixture: dict[str, Path]
) -> list[str]:
    return [
        f"--module={module}",
        "--device=local-task",
        f"--parameters=e5={irpa}",
        "--parameter_mode=preload",
        "--task_topology_group_count=1",
        "--function=sentence_embedding",
        f"--input=@{fixture['ids']}",
        f"--input=@{fixture['mask']}",
        f"--input=@{fixture['types']}",
    ]


def check_numerics(
    runner: str,
    artifact: CompiledArtifact,
    length: int,
    irpa: Path,
    fixture: dict[str, Path],
    output_dir: Path,
    cpu: int | None,
) -> None:
    command = [
        runner,
        *runtime_arguments(artifact.vmfb, irpa, fixture),
        f"--expected_output=@{fixture['expected']}",
        "--expected_f32_threshold=0.0002",
    ]
    label = f"s{length}-{artifact.variant}"
    run_command(
        command,
        cpu,
        output_dir / f"{label}.check.stdout",
        output_dir / f"{label}.check.stderr",
    )


def parse_allocator_statistics(stderr: str) -> tuple[int, int]:
    host = re.search(r"HOST_LOCAL:\s+(\d+)B peak", stderr)
    device = re.search(r"DEVICE_LOCAL:\s+(\d+)B peak", stderr)
    if host is None or device is None:
        raise AssertionError("IREE benchmark omitted allocator peak statistics")
    return int(host.group(1)), int(device.group(1))


def parse_trials(stdout: str) -> list[float]:
    document = json.loads(stdout)
    trials = []
    for record in document["benchmarks"]:
        if record["run_type"] != "iteration":
            continue
        trials.append(float(record["real_time"]) * TIME_TO_MS[record["time_unit"]])
    if not trials:
        raise AssertionError("IREE benchmark returned no repeated trials")
    return trials


def probe_memory(
    args: argparse.Namespace,
    artifact: CompiledArtifact,
    length: int,
    irpa: Path,
    fixture: dict[str, Path],
    output_dir: Path,
    cpu: int | None,
) -> dict[str, int | float]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--memory-probe",
        "--module",
        str(artifact.vmfb),
        "--parameters",
        str(irpa),
        "--ids",
        str(fixture["ids"]),
        "--mask",
        str(fixture["mask"]),
        "--types",
        str(fixture["types"]),
        "--memory-warmup",
        str(args.memory_warmup),
    ]
    label = f"s{length}-{artifact.variant}.memory"
    completed = run_command(
        command,
        cpu,
        output_dir / f"{label}.json",
        output_dir / f"{label}.stderr",
    )
    return json.loads(completed.stdout)


def benchmark(
    args: argparse.Namespace,
    tool: str,
    artifact: CompiledArtifact,
    length: int,
    irpa: Path,
    fixture: dict[str, Path],
    output_dir: Path,
    cpu: int | None,
    canonical_parameter_bytes: int,
) -> dict[str, object]:
    command = [
        tool,
        *runtime_arguments(artifact.vmfb, irpa, fixture),
        f"--benchmark_min_time={args.benchmark_min_time}",
        f"--benchmark_min_warmup_time={args.warmup_seconds}",
        f"--benchmark_repetitions={args.repeats}",
        "--benchmark_report_aggregates_only=false",
        "--benchmark_format=json",
        "--print_statistics=true",
    ]
    label = f"s{length}-{artifact.variant}"
    completed = run_command(
        command,
        cpu,
        output_dir / f"{label}.raw.json",
        output_dir / f"{label}.allocator.txt",
    )
    trials = parse_trials(completed.stdout)
    host_peak, device_peak = parse_allocator_statistics(completed.stderr)
    warmed = probe_memory(args, artifact, length, irpa, fixture, output_dir, cpu)
    return {
        "length": length,
        "variant": artifact.variant,
        "trials_ms": trials,
        "median_ms": statistics.median(trials),
        "mean_ms": statistics.mean(trials),
        "stdev_ms": statistics.stdev(trials) if len(trials) > 1 else 0.0,
        "min_ms": min(trials),
        "max_ms": max(trials),
        "warmed_rss_bytes": warmed["warmed_rss_bytes"],
        "process_high_water_bytes": warmed["process_high_water_bytes"],
        "host_local_peak_bytes": host_peak,
        "device_local_peak_bytes": device_peak,
        "active_scratch_bytes": None,
        "active_scratch_upper_bound_bytes": device_peak,
        "device_peak_minus_canonical_parameter_bytes": max(
            0, device_peak - canonical_parameter_bytes
        ),
    }


def rss_bytes() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, value, unit = line.split()
            if unit != "kB":
                raise AssertionError(f"unexpected RSS unit {unit}")
            values[name.rstrip(":")] = int(value) * 1024
    return values["VmRSS"], values["VmHWM"]


def memory_probe(args: argparse.Namespace) -> int:
    import iree.runtime as ireert

    required = (args.module, args.parameters, args.ids, args.mask, args.types)
    if any(path is None for path in required):
        raise ValueError("internal memory probe is missing an artifact path")
    ireert.flags.parse_flags("--task_topology_group_count=1")
    config = ireert.Config("local-task")
    index = ireert.ParameterIndex()
    index.load(str(args.parameters))
    parameter_module = ireert.create_io_parameters_module(
        config.vm_instance, index.create_provider(scope="e5")
    )
    module = ireert.VmModule.mmap(config.vm_instance, str(args.module))
    context = ireert.SystemContext(
        vm_modules=(parameter_module, module), config=config
    )
    function = context.modules.module.sentence_embedding
    inputs = [np.load(path) for path in (args.ids, args.mask, args.types)]
    result = None
    for _ in range(args.memory_warmup):
        result = np.asarray(function(*inputs))
    if result is None:
        raise ValueError("--memory-warmup must be positive")
    rss, high_water = rss_bytes()
    print(
        json.dumps(
            {
                "warmed_rss_bytes": rss,
                "process_high_water_bytes": high_water,
                "checksum": float(result.sum()),
            }
        )
    )
    return 0


def artifact_record(artifact: CompiledArtifact) -> dict[str, object]:
    return {
        "name": artifact.name,
        "variant": artifact.variant,
        "source": str(artifact.source),
        "compile_s": artifact.compile_s,
        "oracle_frontend_s": artifact.oracle_frontend_s,
        "vmfb_bytes": artifact.vmfb_bytes,
        "vm_flatbuffer_bytes": artifact.vm_flatbuffer_bytes,
        "vm_bytecode_bytes": artifact.vm_bytecode_bytes,
        "embedded_executable_bytes": artifact.embedded_executable_bytes,
        "variant_metadata_bytes": None,
    }


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            return match.group(1)
    return platform.processor() or "unknown"


def main() -> int:
    args = parse_args()
    if args.memory_probe:
        return memory_probe(args)

    lengths = parse_lengths(args.lengths)
    variants = parse_variants(args.variants)
    unsupported_dispatch_lengths = set(lengths) - set(EXACT_LENGTHS)
    if "dispatched" in variants and unsupported_dispatch_lengths:
        raise ValueError(
            "dispatched benchmark lengths must be configured exact lengths; got "
            f"{sorted(unsupported_dispatch_lengths)}"
        )
    if args.repeats < 2 or args.warmup_seconds <= 0 or args.memory_warmup <= 0:
        raise ValueError("repeats must be >=2 and warmups must be positive")
    cpu = select_cpu(args)
    output_dir = make_output_dir(args)
    for path in (
        args.artifact_dir / "sentence_embedding.onnx",
        args.artifact_dir / "e5.irpa",
        args.artifact_dir / "sentence_embedding.core.mlir",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run check-e5 first")
    if "dispatched" in variants and not (
        args.artifact_dir / "sentence_embedding.multiversioned.mlir"
    ).is_file():
        raise FileNotFoundError("missing multiversioned core; run check-e5 first")

    compiler = require_tool(args.iree_compile)
    benchmark_tool = require_tool(args.iree_benchmark)
    runner = require_tool(args.iree_run)
    module_dumper = require_tool(args.iree_dump_module)
    parameter_dumper = require_tool(args.iree_dump_parameters)
    validate_bridge_artifacts(args, variants, output_dir, cpu)
    accounting = parameter_accounting(
        parameter_dumper, args.artifact_dir / "e5.irpa", cpu
    )
    if accounting["irpa_entries"] != 198:
        raise AssertionError("benchmark requires the canonical 198-entry E5 archive")

    oracle_times = {}
    if "static-oracle" in variants:
        oracle_times = build_static_oracles(args, lengths, output_dir, cpu)

    artifacts: dict[str, CompiledArtifact] = {}
    if "dynamic" in variants:
        artifacts["dynamic"] = compile_artifact(
            "dynamic",
            "dynamic",
            args.artifact_dir / "sentence_embedding.core.mlir",
            output_dir,
            compiler,
            module_dumper,
            cpu,
        )
    if "dispatched" in variants:
        artifacts["dispatched"] = compile_artifact(
            "dispatched",
            "dispatched",
            args.artifact_dir / "sentence_embedding.multiversioned.mlir",
            output_dir,
            compiler,
            module_dumper,
            cpu,
        )
    if "static-oracle" in variants:
        for length in lengths:
            key = f"static-oracle-s{length}"
            artifacts[key] = compile_artifact(
                key,
                "static-oracle",
                args.artifact_dir
                / f"sentence_embedding.static_s{length}.core.mlir",
                output_dir,
                compiler,
                module_dumper,
                cpu,
                oracle_times[length],
            )

    fixtures = make_inputs(args.artifact_dir, output_dir, lengths)
    measurements = []
    for length in lengths:
        for variant in variants:
            key = f"static-oracle-s{length}" if variant == "static-oracle" else variant
            artifact = artifacts[key]
            check_numerics(
                runner,
                artifact,
                length,
                args.artifact_dir / "e5.irpa",
                fixtures[length],
                output_dir,
                cpu,
            )
            measurements.append(
                benchmark(
                    args,
                    benchmark_tool,
                    artifact,
                    length,
                    args.artifact_dir / "e5.irpa",
                    fixtures[length],
                    output_dir,
                    cpu,
                    accounting["canonical_parameter_bytes"],
                )
            )

    summary = {
        "environment": {
            "cpu": cpu_model(),
            "affinity": "off" if cpu is None else f"cpu{cpu}",
            "platform": platform.platform(),
            "iree_compile": run_command([compiler, "--version"], cpu).stdout.strip(),
        },
        "configuration": {
            "lengths": lengths,
            "variants": variants,
            "repeats": args.repeats,
            "benchmark_min_time": args.benchmark_min_time,
            "warmup_seconds": args.warmup_seconds,
            "memory_warmup_invocations": args.memory_warmup,
            "iree_flags": list(IREE_FLAGS),
            "runtime_device": "local-task",
            "runtime_workers": 1,
            "parameter_mode": "preload",
            "warmed_rss_probe": (
                "separate pinned Python IREE-runtime process after warmup; includes "
                "Python/binding overhead and a memory-mapped ParameterIndex provider"
            ),
        },
        "boundaries": {
            "dispatch_in_dispatched_latency": True,
            "tokenization_in_latency": False,
            "caller_padding_in_latency": False,
            "fixture_file_io_in_latency": False,
            "parameter_loading_in_latency": False,
        },
        "accounting": {
            **accounting,
            "duplicated_parameter_bytes": 0,
            "length_specific_parameter_bytes": 0,
            "prepacked_weight_bytes": None,
            "prepacked_weight_note": (
                "No second archive or serialized learned payload exists, but "
                "runtime/compiler prepacking is not separately exposed."
            ),
            "variant_metadata_bytes": None,
            "variant_metadata_note": (
                "Variant functions, guards, and symbols are included in VM bytecode "
                "and VMFB totals but are not separately isolated."
            ),
            "active_scratch_bytes": None,
            "active_scratch_note": (
                "IREE allocator statistics do not separate scratch from I/O and "
                "other transient device allocations. DEVICE_LOCAL peak is reported "
                "as a strict upper bound. The arithmetic residual after subtracting "
                "canonical bytes may still include prepacked/derived parameter "
                "layouts and is not an exact scratch or non-parameter measurement."
            ),
        },
        "artifacts": [artifact_record(item) for item in artifacts.values()],
        "measurements": measurements,
        "conclusion": (
            "Descriptive measurements only; no causal speedup claim is made without "
            "separate IR, generated-code, or counter evidence."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"E5 benchmark results: {output_dir}")
    print(
        f"affinity={summary['environment']['affinity']} repeats={args.repeats} "
        f"warmup_seconds={args.warmup_seconds}"
    )
    print("variant         S   median_ms     min_ms     max_ms  warmed_rss")
    for item in measurements:
        print(
            f"{item['variant']:<15}{item['length']:>3}"
            f"{item['median_ms']:>12.4f}{item['min_ms']:>11.4f}"
            f"{item['max_ms']:>11.4f}{item['warmed_rss_bytes']:>13}"
        )
    print(
        f"canonical_parameter_bytes={accounting['canonical_parameter_bytes']} "
        "duplicated_parameter_bytes=0 prepacked_weight_bytes=not_isolated"
    )
    print("active_scratch_bytes=not_isolated; DEVICE_LOCAL peak is the upper bound")
    print("No causal performance claim is made by this harness.")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
