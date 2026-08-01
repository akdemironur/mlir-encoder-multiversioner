#!/usr/bin/env python3
"""Matched controls for attributing E5 latency changes to static clones."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from bench_e5_iree import (  # noqa: E402
    IREE_FLAGS,
    make_inputs,
    parameter_accounting,
    parse_allocator_statistics,
    parse_module_metadata,
    parse_trials,
    require_tool,
    rss_bytes,
    run_command,
    select_cpu,
)
from e5_common import DEFAULT_ARTIFACT_DIR, sha256_file  # noqa: E402

TARGET_LENGTHS = (256, 512)
FALLBACK_LENGTH = 257
PERF_EVENTS = "cycles,instructions,branches,branch-misses,cache-misses"


@dataclass(frozen=True)
class PathSpec:
    length: int
    path: str
    module: Path
    function: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--benchmark-min-time", default="0.05s")
    parser.add_argument("--warmup-seconds", type=float, default=0.2)
    parser.add_argument("--memory-warmup", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--equivalence-margin-percent", type=float, default=1.0)
    parser.add_argument("--collect-perf", action="store_true")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--no-affinity", action="store_true")
    parser.add_argument(
        "--mlir-opt", type=Path, default=REPO_ROOT / "build/llvm/bin/mlir-opt"
    )
    parser.add_argument(
        "--plugin",
        type=Path,
        default=REPO_ROOT / "build/shortseq-pinned/lib/ShortSeqPasses.so",
    )
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--iree-benchmark", default="iree-benchmark-module")
    parser.add_argument("--iree-run", default="iree-run-module")
    parser.add_argument("--iree-dump-module", default="iree-dump-module")
    parser.add_argument("--iree-dump-parameters", default="iree-dump-parameters")

    parser.add_argument("--memory-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parameters", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--function", help=argparse.SUPPRESS)
    parser.add_argument("--ids", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mask", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--types", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = args.artifact_dir / "benchmarks" / f"{stamp}-static-causality"
    else:
        output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def build_multiversioned(
    args: argparse.Namespace,
    output: Path,
    name: str,
    expose_static_variants: bool,
    cpu: int | None,
) -> Path:
    transformed = output / f"{name}.mlir"
    options = "lengths=256,512"
    if expose_static_variants:
        options += " expose-static-variants=true"
    pipeline = f"builtin.module(shortseq-specialize{{{options}}})"
    run_command(
        [
            str(args.mlir_opt),
            "--allow-unregistered-dialect",
            f"--load-pass-plugin={args.plugin}",
            f"--pass-pipeline={pipeline}",
            str(args.artifact_dir / "sentence_embedding.core.mlir"),
            "-o",
            str(transformed),
        ],
        cpu,
        output / f"{name}.transform.stdout",
        output / f"{name}.transform.stderr",
    )
    run_command(
        [
            str(args.mlir_opt),
            "--allow-unregistered-dialect",
            str(transformed),
            "-o",
            os.devnull,
        ],
        cpu,
        output / f"{name}.verify.stdout",
        output / f"{name}.verify.stderr",
    )
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/check_e5_bridge.py"),
            "--artifact-dir",
            str(args.artifact_dir),
            "--core",
            str(transformed),
            "--specialized-lengths",
            "256,512",
        ],
        cpu,
        output / f"{name}.artifact-check.stdout",
        output / f"{name}.artifact-check.stderr",
    )
    return transformed


def compile_module(
    compiler: str,
    dumper: str,
    source: Path,
    name: str,
    output: Path,
    cpu: int | None,
) -> dict[str, object]:
    dump_root = output / f"{name}-compiler-evidence"
    phases = dump_root / "phases"
    executable = dump_root / "executable"
    phases.mkdir(parents=True)
    executable.mkdir(parents=True)
    vmfb = output / f"{name}.vmfb"
    scheduling_stats = dump_root / "scheduling.json"
    command = [
        compiler,
        str(source),
        *IREE_FLAGS,
        f"--dump-compilation-phases-to={phases}",
        f"--iree-hal-dump-executable-files-to={executable}",
        f"--iree-scheduling-dump-statistics-file={scheduling_stats}",
        "--iree-scheduling-dump-statistics-format=json",
        "-o",
        str(vmfb),
    ]
    start = time.perf_counter()
    run_command(
        command,
        cpu,
        output / f"{name}.compile.stdout",
        output / f"{name}.compile.stderr",
    )
    compile_s = time.perf_counter() - start
    metadata = run_command([dumper, "--output=metadata", str(vmfb)], cpu).stdout
    disassembly = run_command([dumper, "--output=disassembly", str(vmfb)], cpu).stdout
    (output / f"{name}.module-metadata.txt").write_text(metadata, encoding="utf-8")
    (output / f"{name}.vm-disassembly.mlir").write_text(disassembly, encoding="utf-8")
    flatbuffer, bytecode, executable_bytes = parse_module_metadata(metadata)
    return {
        "name": name,
        "source": str(source),
        "vmfb": str(vmfb),
        "compile_s": compile_s,
        "vmfb_bytes": vmfb.stat().st_size,
        "vm_flatbuffer_bytes": flatbuffer,
        "vm_bytecode_bytes": bytecode,
        "embedded_executable_bytes": executable_bytes,
        "source_sha256": sha256_file(source),
        "compiler_evidence": str(dump_root),
        "scheduling_statistics": str(scheduling_stats),
    }


def check_exports(metadata: Path, expect_static: bool) -> None:
    text = metadata.read_text(encoding="utf-8")
    expected = ["sentence_embedding_generic(", "sentence_embedding("]
    if expect_static:
        expected.extend(("sentence_embedding_s256(", "sentence_embedding_s512("))
    missing = [symbol for symbol in expected if symbol not in text]
    if missing:
        raise AssertionError(f"multiversioned VMFB omitted exports: {missing}")
    for symbol in ("sentence_embedding_s256(", "sentence_embedding_s512("):
        if not expect_static and symbol in text:
            raise AssertionError(f"production VMFB unexpectedly exported {symbol}")


def runtime_arguments(
    spec: PathSpec, parameters: Path, fixture: dict[str, Path]
) -> list[str]:
    return [
        f"--module={spec.module}",
        "--device=local-task",
        f"--parameters=e5={parameters}",
        "--parameter_mode=preload",
        "--task_topology_group_count=1",
        f"--function={spec.function}",
        f"--input=@{fixture['ids']}",
        f"--input=@{fixture['mask']}",
        f"--input=@{fixture['types']}",
    ]


def check_numerics(
    runner: str,
    spec: PathSpec,
    parameters: Path,
    fixture: dict[str, Path],
    output: Path,
    cpu: int | None,
) -> None:
    label = f"s{spec.length}-{spec.path}"
    run_command(
        [
            runner,
            *runtime_arguments(spec, parameters, fixture),
            f"--expected_output=@{fixture['expected']}",
            "--expected_f32_threshold=0.0002",
        ],
        cpu,
        output / f"{label}.check.stdout",
        output / f"{label}.check.stderr",
    )


def run_trial(
    benchmark_tool: str,
    spec: PathSpec,
    parameters: Path,
    fixture: dict[str, Path],
    args: argparse.Namespace,
    output: Path,
    session: int,
    cpu: int | None,
) -> tuple[float, int, int]:
    label = f"session-{session:02d}-s{spec.length}-{spec.path}"
    completed = run_command(
        [
            benchmark_tool,
            *runtime_arguments(spec, parameters, fixture),
            f"--benchmark_min_time={args.benchmark_min_time}",
            f"--benchmark_min_warmup_time={args.warmup_seconds}",
            "--benchmark_repetitions=1",
            "--benchmark_report_aggregates_only=false",
            "--benchmark_format=json",
            "--print_statistics=true",
        ],
        cpu,
        output / f"{label}.raw.json",
        output / f"{label}.allocator.txt",
    )
    trials = parse_trials(completed.stdout)
    if len(trials) != 1:
        raise AssertionError(f"{label} returned {len(trials)} trials, expected one")
    host_peak, device_peak = parse_allocator_statistics(completed.stderr)
    return trials[0], host_peak, device_peak


def bootstrap_median_ci(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    if len(values) < 2 or samples < 100:
        raise ValueError("bootstrap CI requires at least two values and 100 samples")
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return medians[int(samples * 0.025)], medians[int(samples * 0.975)]


def effect_record(
    records: dict[tuple[int, str], list[float]],
    length: int,
    reference: str,
    candidate: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    baseline = records[(length, reference)]
    changed = records[(length, candidate)]
    if len(baseline) != len(changed):
        raise AssertionError("paired paths have different session counts")
    reductions = [
        100.0 * (reference_ms - candidate_ms) / reference_ms
        for reference_ms, candidate_ms in zip(baseline, changed, strict=True)
    ]
    lower, upper = bootstrap_median_ci(reductions, samples, seed)
    return {
        "length": length,
        "reference": reference,
        "candidate": candidate,
        "reference_median_ms": statistics.median(baseline),
        "candidate_median_ms": statistics.median(changed),
        "median_paired_reduction_percent": statistics.median(reductions),
        "bootstrap_95_percent_ci": [lower, upper],
    }


def positive_effect_supported(effect: dict[str, object]) -> bool:
    return effect["bootstrap_95_percent_ci"][0] > 0


def equivalent_effect_supported(
    effect: dict[str, object], margin_percent: float
) -> bool:
    lower, upper = effect["bootstrap_95_percent_ci"]
    return lower >= -margin_percent and upper <= margin_percent


def parse_cpu_list(text: str) -> list[int]:
    cpus = []
    for item in text.split(","):
        bounds = [int(value) for value in item.split("-")]
        if len(bounds) == 1:
            cpus.append(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            cpus.extend(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError(f"invalid Linux CPU list: {text!r}")
    return cpus


def read_cpu_control(cpu: int | None) -> dict[str, object]:
    if cpu is None:
        return {
            "controlled": False,
            "frequency_and_turbo_controlled": False,
            "smt_sibling_isolation_verified": False,
            "reason": "CPU affinity disabled",
        }
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}")

    def read(path: Path) -> str | None:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    governor = read(root / "cpufreq/scaling_governor")
    minimum = read(root / "cpufreq/scaling_min_freq")
    maximum = read(root / "cpufreq/scaling_max_freq")
    siblings = read(root / "topology/thread_siblings_list")
    no_turbo = read(Path("/sys/devices/system/cpu/intel_pstate/no_turbo"))
    fixed_frequency = minimum is not None and minimum == maximum
    turbo_disabled = no_turbo == "1"
    sibling_cpus = parse_cpu_list(siblings) if siblings is not None else []
    online_siblings = []
    for sibling in sibling_cpus:
        if sibling == cpu:
            continue
        online = read(Path(f"/sys/devices/system/cpu/cpu{sibling}/online"))
        if online != "0":
            online_siblings.append(sibling)
    smt_isolation_verified = siblings is not None and not online_siblings
    frequency_and_turbo_controlled = fixed_frequency and turbo_disabled
    return {
        "controlled": frequency_and_turbo_controlled and smt_isolation_verified,
        "frequency_and_turbo_controlled": frequency_and_turbo_controlled,
        "smt_sibling_isolation_verified": smt_isolation_verified,
        "governor": governor,
        "scaling_min_freq": minimum,
        "scaling_max_freq": maximum,
        "intel_pstate_no_turbo": no_turbo,
        "thread_siblings": siblings,
        "online_siblings": online_siblings,
        "frequency_note": (
            "frequency control requires fixed min/max frequency and disabled turbo"
        ),
        "smt_note": (
            "SMT isolation is verified only when the selected logical CPU has no "
            "sibling or every sibling is offline; external idleness is not inferred"
        ),
    }


def probe_memory(
    spec: PathSpec,
    parameters: Path,
    fixture: dict[str, Path],
    args: argparse.Namespace,
    output: Path,
    cpu: int | None,
) -> dict[str, object]:
    label = f"s{spec.length}-{spec.path}.memory"
    completed = run_command(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--memory-probe",
            "--module",
            str(spec.module),
            "--parameters",
            str(parameters),
            "--function",
            spec.function,
            "--ids",
            str(fixture["ids"]),
            "--mask",
            str(fixture["mask"]),
            "--types",
            str(fixture["types"]),
            "--memory-warmup",
            str(args.memory_warmup),
        ],
        cpu,
        output / f"{label}.json",
        output / f"{label}.stderr",
    )
    return json.loads(completed.stdout)


def memory_probe(args: argparse.Namespace) -> int:
    import iree.runtime as ireert

    required = (
        args.module,
        args.parameters,
        args.function,
        args.ids,
        args.mask,
        args.types,
    )
    if any(item is None for item in required):
        raise ValueError("internal memory probe is missing an argument")
    ireert.flags.parse_flags("--task_topology_group_count=1")
    config = ireert.Config("local-task")
    index = ireert.ParameterIndex()
    index.load(str(args.parameters))
    parameter_module = ireert.create_io_parameters_module(
        config.vm_instance, index.create_provider(scope="e5")
    )
    module = ireert.VmModule.mmap(config.vm_instance, str(args.module))
    context = ireert.SystemContext(vm_modules=(parameter_module, module), config=config)
    function = getattr(context.modules.module, args.function)
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


def perf_available(cpu: int | None) -> tuple[bool, str]:
    perf = shutil.which("perf")
    if perf is None:
        return False, "perf not found"
    completed = subprocess.run(
        [perf, "stat", "-x,", "-e", "cycles", "true"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=(None if cpu is None else lambda: os.sched_setaffinity(0, {cpu})),
    )
    return completed.returncode == 0, completed.stderr.strip()


def collect_perf(
    benchmark_tool: str,
    specs: list[PathSpec],
    parameters: Path,
    fixtures: dict[int, dict[str, Path]],
    args: argparse.Namespace,
    output: Path,
    cpu: int | None,
) -> dict[str, object]:
    available, diagnostic = perf_available(cpu)
    if not available:
        return {"available": False, "diagnostic": diagnostic}
    perf = require_tool("perf")
    files = []
    for spec in specs:
        label = f"s{spec.length}-{spec.path}.perf"
        command = [
            perf,
            "stat",
            "-x,",
            "-e",
            PERF_EVENTS,
            "--",
            benchmark_tool,
            *runtime_arguments(spec, parameters, fixtures[spec.length]),
            f"--benchmark_min_time={args.benchmark_min_time}",
            f"--benchmark_min_warmup_time={args.warmup_seconds}",
            "--benchmark_repetitions=1",
            "--benchmark_report_aggregates_only=false",
            "--benchmark_format=json",
        ]
        run_command(
            command,
            cpu,
            output / f"{label}.raw.json",
            output / f"{label}.txt",
        )
        files.append(str(output / f"{label}.txt"))
    return {
        "available": True,
        "one_shot_exploratory": True,
        "supports_causal_mechanism_claim": False,
        "events": PERF_EVENTS.split(","),
        "scope": (
            "one deterministic, unpaired whole-process run per path, including "
            "initialization and warmup"
        ),
        "raw_files": files,
    }


def main() -> int:
    args = parse_args()
    if args.memory_probe:
        return memory_probe(args)
    if args.sessions < 2 or args.bootstrap_samples < 100:
        raise ValueError("sessions must be >=2 and bootstrap samples must be >=100")
    if args.equivalence_margin_percent <= 0:
        raise ValueError("equivalence margin must be positive")
    if args.warmup_seconds <= 0 or args.memory_warmup <= 0:
        raise ValueError("warmups must be positive")
    for path in (
        args.artifact_dir / "sentence_embedding.onnx",
        args.artifact_dir / "sentence_embedding.core.mlir",
        args.artifact_dir / "e5.irpa",
        args.mlir_opt,
        args.plugin,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing required artifact: {path}")

    cpu = select_cpu(args)
    output = make_output_dir(args)
    compiler = require_tool(args.iree_compile)
    benchmark_tool = require_tool(args.iree_benchmark)
    runner = require_tool(args.iree_run)
    module_dumper = require_tool(args.iree_dump_module)
    parameter_dumper = require_tool(args.iree_dump_parameters)
    parameters = args.artifact_dir / "e5.irpa"

    dynamic_source = output / "dynamic.mlir"
    shutil.copy2(args.artifact_dir / "sentence_embedding.core.mlir", dynamic_source)
    production_source = build_multiversioned(
        args, output, "multiversioned-production", False, cpu
    )
    observable_source = build_multiversioned(
        args, output, "multiversioned-observable", True, cpu
    )
    artifacts = [
        compile_module(compiler, module_dumper, dynamic_source, "dynamic", output, cpu),
        compile_module(
            compiler,
            module_dumper,
            production_source,
            "multiversioned-production",
            output,
            cpu,
        ),
        compile_module(
            compiler,
            module_dumper,
            observable_source,
            "multiversioned-observable",
            output,
            cpu,
        ),
    ]
    dynamic_vmfb = Path(artifacts[0]["vmfb"])
    production_vmfb = Path(artifacts[1]["vmfb"])
    observable_vmfb = Path(artifacts[2]["vmfb"])
    check_exports(output / "multiversioned-production.module-metadata.txt", False)
    check_exports(output / "multiversioned-observable.module-metadata.txt", True)

    accounting = parameter_accounting(parameter_dumper, parameters, cpu)
    if accounting["irpa_entries"] != 198:
        raise AssertionError("causality experiment requires the 198-entry E5 IRPA")

    specs = []
    for length in TARGET_LENGTHS:
        specs.extend(
            (
                PathSpec(length, "dynamic", dynamic_vmfb, "sentence_embedding"),
                PathSpec(
                    length,
                    "production-generic",
                    production_vmfb,
                    "sentence_embedding_generic",
                ),
                PathSpec(
                    length,
                    "production-dispatched",
                    production_vmfb,
                    "sentence_embedding",
                ),
                PathSpec(
                    length,
                    "observable-generic",
                    observable_vmfb,
                    "sentence_embedding_generic",
                ),
                PathSpec(
                    length,
                    "direct-static-diagnostic",
                    observable_vmfb,
                    f"sentence_embedding_s{length}",
                ),
                PathSpec(
                    length,
                    "observable-dispatched",
                    observable_vmfb,
                    "sentence_embedding",
                ),
            )
        )
    specs.extend(
        (
            PathSpec(
                FALLBACK_LENGTH,
                "generic-fallback-control",
                production_vmfb,
                "sentence_embedding_generic",
            ),
            PathSpec(
                FALLBACK_LENGTH,
                "dispatched-fallback",
                production_vmfb,
                "sentence_embedding",
            ),
        )
    )

    fixtures = make_inputs(
        args.artifact_dir, output, [*TARGET_LENGTHS, FALLBACK_LENGTH]
    )
    for spec in specs:
        check_numerics(runner, spec, parameters, fixtures[spec.length], output, cpu)

    records: dict[tuple[int, str], list[float]] = {
        (spec.length, spec.path): [] for spec in specs
    }
    allocator: dict[tuple[int, str], list[tuple[int, int]]] = {
        key: [] for key in records
    }
    generator = random.Random(args.seed)
    run_order = []
    for session in range(args.sessions):
        session_specs = list(specs)
        generator.shuffle(session_specs)
        run_order.append([f"s{spec.length}-{spec.path}" for spec in session_specs])
        for spec in session_specs:
            trial, host_peak, device_peak = run_trial(
                benchmark_tool,
                spec,
                parameters,
                fixtures[spec.length],
                args,
                output,
                session,
                cpu,
            )
            key = (spec.length, spec.path)
            records[key].append(trial)
            allocator[key].append((host_peak, device_peak))

    measurements = []
    for spec in specs:
        key = (spec.length, spec.path)
        values = records[key]
        memory = probe_memory(
            spec, parameters, fixtures[spec.length], args, output, cpu
        )
        measurements.append(
            {
                "length": spec.length,
                "path": spec.path,
                "module": str(spec.module),
                "function": spec.function,
                "fresh_process_trials_ms": values,
                "median_ms": statistics.median(values),
                "mean_ms": statistics.mean(values),
                "stdev_ms": statistics.stdev(values),
                "host_local_peak_bytes": statistics.median(
                    item[0] for item in allocator[key]
                ),
                "device_local_peak_bytes": statistics.median(
                    item[1] for item in allocator[key]
                ),
                **memory,
            }
        )

    effects = []
    effect_seed = args.seed + 1
    for length in TARGET_LENGTHS:
        for reference, candidate in (
            ("production-generic", "production-dispatched"),
            ("observable-generic", "direct-static-diagnostic"),
            ("direct-static-diagnostic", "observable-dispatched"),
            ("dynamic", "production-generic"),
            ("production-generic", "observable-generic"),
            ("production-dispatched", "observable-dispatched"),
        ):
            effects.append(
                effect_record(
                    records,
                    length,
                    reference,
                    candidate,
                    args.bootstrap_samples,
                    effect_seed,
                )
            )
            effect_seed += 1
    effects.append(
        effect_record(
            records,
            FALLBACK_LENGTH,
            "generic-fallback-control",
            "dispatched-fallback",
            args.bootstrap_samples,
            effect_seed,
        )
    )

    cpu_control = read_cpu_control(cpu)
    effect_index = {
        (item["length"], item["reference"], item["candidate"]): item for item in effects
    }

    def effect(length: int, reference: str, candidate: str) -> dict[str, object]:
        return effect_index[(length, reference, candidate)]

    speedup_checks = {
        f"s{length}_production_static": positive_effect_supported(
            effect(length, "production-generic", "production-dispatched")
        )
        for length in TARGET_LENGTHS
    }
    speedup_checks.update(
        {
            f"s{length}_direct_static_diagnostic": positive_effect_supported(
                effect(length, "observable-generic", "direct-static-diagnostic")
            )
            for length in TARGET_LENGTHS
        }
    )
    margin = args.equivalence_margin_percent
    equivalence_checks = {
        f"s{length}_diagnostic_dispatch_overhead": equivalent_effect_supported(
            effect(
                length,
                "direct-static-diagnostic",
                "observable-dispatched",
            ),
            margin,
        )
        for length in TARGET_LENGTHS
    }
    equivalence_checks.update(
        {
            f"s{length}_dynamic_module_layout": equivalent_effect_supported(
                effect(length, "dynamic", "production-generic"), margin
            )
            for length in TARGET_LENGTHS
        }
    )
    equivalence_checks.update(
        {
            f"s{length}_generic_visibility_observer": equivalent_effect_supported(
                effect(length, "production-generic", "observable-generic"), margin
            )
            for length in TARGET_LENGTHS
        }
    )
    equivalence_checks.update(
        {
            f"s{length}_dispatched_visibility_observer": equivalent_effect_supported(
                effect(
                    length,
                    "production-dispatched",
                    "observable-dispatched",
                ),
                margin,
            )
            for length in TARGET_LENGTHS
        }
    )
    equivalence_checks["s257_guarded_fallback"] = equivalent_effect_supported(
        effect(
            FALLBACK_LENGTH,
            "generic-fallback-control",
            "dispatched-fallback",
        ),
        margin,
    )
    static_speedup_signals = all(speedup_checks.values())
    timing_signature = static_speedup_signals and all(equivalence_checks.values())
    perf = (
        collect_perf(benchmark_tool, specs, parameters, fixtures, args, output, cpu)
        if args.collect_perf
        else {
            "available": False,
            "diagnostic": "not requested",
            "supports_causal_mechanism_claim": False,
        }
    )
    summary = {
        "experiment": "matched E5 static-path causality controls",
        "configuration": {
            "target_lengths": list(TARGET_LENGTHS),
            "fallback_length": FALLBACK_LENGTH,
            "sessions": args.sessions,
            "random_seed": args.seed,
            "randomized_order": True,
            "one_fresh_process_per_path_per_session": True,
            "benchmark_min_time": args.benchmark_min_time,
            "warmup_seconds": args.warmup_seconds,
            "equivalence_margin_percent": margin,
            "iree_flags": list(IREE_FLAGS),
            "cpu": cpu,
            "cpu_control": cpu_control,
        },
        "boundaries": {
            "timed": "warmed IREE invocation loop",
            "dispatch_in_dispatched_paths": True,
            "tokenization": False,
            "caller_padding": False,
            "fixture_io": False,
            "parameter_loading": False,
            "perf_scope": perf.get("scope"),
        },
        "controls": {
            "dynamic": "original audited core in its own module",
            "production-generic": (
                "unchanged generic body in the production-topology VMFB"
            ),
            "production-dispatched": (
                "the real wrapper selecting private pass-produced clones"
            ),
            "observable-generic": (
                "unchanged generic body in a diagnostic VMFB with public clones"
            ),
            "direct-static-diagnostic": (
                "the exact public clone; diagnostic because visibility changes topology"
            ),
            "observable-dispatched": (
                "the wrapper in the diagnostic public-clone VMFB"
            ),
            "dispatched-fallback": (
                "the same wrapper at unsupported S=257 selecting the unchanged generic body"
            ),
        },
        "accounting": {
            **accounting,
            "parameter_archives": 1,
            "duplicated_parameter_bytes": 0,
            "length_specific_parameter_bytes": 0,
            "prepacked_weight_bytes": None,
            "active_scratch_bytes": None,
            "active_scratch_note": "DEVICE_LOCAL peak is retained only as an upper bound",
        },
        "artifacts": artifacts,
        "run_order": run_order,
        "measurements": measurements,
        "paired_effects": effects,
        "perf": perf,
        "claim_gate": {
            "static_speedup_signals_supported": static_speedup_signals,
            "timing_signature_supported": timing_signature,
            "speedup_checks": speedup_checks,
            "equivalence_checks": equivalence_checks,
            "cpu_frequency_and_turbo_controlled": cpu_control[
                "frequency_and_turbo_controlled"
            ],
            "smt_sibling_isolation_verified": cpu_control[
                "smt_sibling_isolation_verified"
            ],
            "compiler_evidence_retained": True,
            "compiler_evidence_manually_reviewed": False,
            "perf_mechanism_evidence_ready": False,
            "causal_claim_ready": False,
            "note": (
                "A claim requires every speedup and equivalence check, controlled "
                "frequency/turbo, verified SMT isolation, and manual review of "
                "retained compiler/code evidence."
            ),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"E5 static-path causality experiment: {output}")
    print("S    path                         median_ms")
    for item in measurements:
        print(f"{item['length']:<4} {item['path']:<28} {item['median_ms']:>10.4f}")
    print("paired median reductions (positive favors candidate):")
    for item in effects:
        lower, upper = item["bootstrap_95_percent_ci"]
        print(
            f"S={item['length']} {item['reference']} -> {item['candidate']}: "
            f"{item['median_paired_reduction_percent']:.3f}% "
            f"95% CI [{lower:.3f}, {upper:.3f}]"
        )
    print(f"static_speedup_signals_supported={str(static_speedup_signals).lower()}")
    print(f"full_timing_signature_supported={str(timing_signature).lower()}")
    print(f"cpu_controlled={str(cpu_control['controlled']).lower()}")
    print("causal_claim_ready=false (compiler evidence requires manual review)")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
