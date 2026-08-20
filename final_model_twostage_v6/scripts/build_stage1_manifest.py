#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from two_stage_surrogate.data.sample_manifest import build_stage1_manifest, write_manifest  # noqa: E402


# Load config from disk or cache.
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise RuntimeError(f"Config must be a mapping: {path}")
    return config


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata-only Stage 1 Camden manifest.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage1_manifest_camden.yaml")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if os.environ.get("CAMDEN_JOBS_ROOT"):
        config["jobs_root"] = os.environ["CAMDEN_JOBS_ROOT"]
    if args.out_dir is not None:
        config["out_dir"] = str(args.out_dir)
    if args.limit_jobs is not None:
        config["limit_jobs"] = int(args.limit_jobs)

    manifest = build_stage1_manifest(config)
    summary = {
        "jobs": len(manifest["jobs"]),
        "record_count": manifest["record_count"],
        "patch_shape": manifest["patch_shape"],
        "patch_stride": manifest["patch_stride"],
        "out_dir": config["out_dir"],
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.dry_run:
        return
    json_path, csv_path = write_manifest(manifest, config["out_dir"])
    print(f"Saved manifest: {json_path}", flush=True)
    print(f"Saved preview: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
