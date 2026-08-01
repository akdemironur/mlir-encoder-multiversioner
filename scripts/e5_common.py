"""Constants and small helpers for the fixed E5-small-v2 workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "config/e5-small-v2-artifacts.json"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "results/e5-small-v2"


def load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hashes(directory: Path) -> None:
    manifest = load_manifest()
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file():
            raise AssertionError(f"missing pinned artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(
                f"SHA-256 mismatch for {name}: got {actual}, expected {expected}"
            )
