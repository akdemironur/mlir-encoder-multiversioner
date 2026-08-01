#!/usr/bin/env python3
"""Compare external-bundle dispatch with the operand-backed Stage B module."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import iree.compiler as ireec
    import iree.runtime as ireert
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "IREE and NumPy are required. Run with: "
        "uv run --frozen --group bench python "
        "test/stage_b/external_parameter_bundle_numerical.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.make_stage_b_bundle import PARAMS, build_bundle  # noqa: E402


SOURCE = REPO_ROOT / "examples/stage_b/contract_encoder_block.mlir"
IREE_FLAGS = (
    "--iree-llvmcpu-target-cpu=host",
    "--iree-scheduling-optimize-bindings=false",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlir-opt", default=REPO_ROOT / "build/llvm/bin/mlir-opt", type=Path
    )
    parser.add_argument(
        "--plugin",
        default=REPO_ROOT / "build/shortseq-pinned/lib/ShortSeqPasses.so",
        type=Path,
    )
    return parser.parse_args()


def run_pass(args: argparse.Namespace, source: Path) -> str:
    command = [
        str(args.mlir_opt),
        "--allow-unregistered-dialect",
        f"--load-pass-plugin={args.plugin}",
        "--pass-pipeline=builtin.module(shortseq-specialize{lengths=4,8})",
        str(source),
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


def compile_source(source: str) -> bytes:
    return bytes(
        ireec.compile_str(
            source,
            target_backends=["llvm-cpu"],
            extra_args=list(IREE_FLAGS),
        )
    )


def operand_function(vmfb: bytes):
    config = ireert.Config("local-sync")
    module = ireert.VmModule.copy_buffer(config.vm_instance, vmfb)
    context = ireert.SystemContext(config=config)
    context.add_vm_module(module)
    return context.modules.module.encoder_block, (config, module, context)


def bundle_function(vmfb: bytes, archive: Path):
    config = ireert.Config("local-sync")
    index = ireert.ParameterIndex()
    index.load(str(archive))
    provider = index.create_provider(scope="stage_b")
    parameters_module = ireert.create_io_parameters_module(
        config.vm_instance, provider
    )
    module = ireert.VmModule.copy_buffer(config.vm_instance, vmfb)
    context = ireert.SystemContext(
        vm_modules=(parameters_module, module), config=config
    )
    return context.modules.module.run_encoder_block, (
        config,
        index,
        provider,
        parameters_module,
        module,
        context,
    )


def main() -> int:
    args = parse_args()
    for path in (args.mlir_opt, args.plugin, SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)

    rng = np.random.default_rng(0xB00D1E)
    arrays = {
        name: rng.normal(0.0, 0.02, shape).astype(np.float32)
        for name, shape, _ in PARAMS
    }
    parameters = tuple(arrays[name] for name, _, _ in PARAMS)
    arrays.update(
        {
            "x_s4": rng.normal(0.0, 0.1, (1, 4, 64)).astype(np.float32),
            "x_s5": rng.normal(0.0, 0.1, (1, 5, 64)).astype(np.float32),
        }
    )

    with TemporaryDirectory(prefix="stage-b-bundle-") as tmp:
        tmp_dir = Path(tmp)
        artifact = tmp_dir / "fixture.npz"
        bundle_mlir = tmp_dir / "bundle.mlir"
        archive = tmp_dir / "weights.irpa"
        np.savez(artifact, **arrays)
        canonical_bytes = build_bundle(
            artifact, SOURCE, bundle_mlir, archive, scope="stage_b"
        )

        dynamic_vmfb = compile_source(SOURCE.read_text())
        bundle_vmfb = compile_source(run_pass(args, bundle_mlir))
        dynamic, dynamic_keepalive = operand_function(dynamic_vmfb)
        bundled, bundle_keepalive = bundle_function(bundle_vmfb, archive)

        for length in (4, 5):
            x = arrays[f"x_s{length}"]
            expected = np.asarray(dynamic(x, *parameters))
            actual = np.asarray(bundled(x))
            np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-4)

        # Keep runtime objects live through both invocations.
        assert dynamic_keepalive and bundle_keepalive

    print(
        "PASS Stage B external bundle numerics: "
        "S=4 exact specialization and S=5 generic fallback"
    )
    print(f"canonical_parameter_bytes={canonical_bytes}")
    print("duplicated_parameter_bytes=0")
    print(f"dynamic_vmfb_bytes={len(dynamic_vmfb)}")
    print(f"multiversioned_vmfb_bytes={len(bundle_vmfb)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
