#!/usr/bin/env python3
"""Fetch only the pinned, model-specific E5-small-v2 inputs."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

from e5_common import DEFAULT_ARTIFACT_DIR, load_manifest, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def download(url: str, output: Path, expected: str) -> None:
    if output.is_file() and sha256_file(output) == expected:
        print(f"verified existing {output}")
        return
    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"fetching {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    actual = sha256_file(partial)
    if actual != expected:
        partial.unlink()
        raise AssertionError(
            f"SHA-256 mismatch for {output.name}: got {actual}, expected {expected}"
        )
    os.replace(partial, output)
    print(f"verified {output}")


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    revision = manifest["revision"]
    base = f"https://huggingface.co/{manifest['model']}/resolve/{revision}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        download(f"{base}/{name}", args.output_dir / name, expected)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
