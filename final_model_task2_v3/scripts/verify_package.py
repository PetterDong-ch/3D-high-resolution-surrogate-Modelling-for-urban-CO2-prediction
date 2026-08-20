#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# Compute the SHA-256 checksum of a file.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Entry point for the command-line workflow.
def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify the Task2 V3 reproducibility package.")
    parser.add_argument("--data-root", type=Path, help="Optional external-data root.")
    args = parser.parse_args()
    manifest = json.loads((root / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["checkpoints"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing checkpoint: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
        else:
            print(f"OK checkpoint {relative}")
    if args.data_root:
        for relative in manifest["external_data_entries"]:
            if not (args.data_root / relative).exists():
                failures.append(f"missing external data: {relative}")
            else:
                print(f"OK data {relative}")
    else:
        print("External data not checked; pass --data-root after obtaining it.")
    print(f"Python runtime: {sys.version.split()[0]} (reference {manifest['python']})")
    if failures:
        print("Verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Package verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
