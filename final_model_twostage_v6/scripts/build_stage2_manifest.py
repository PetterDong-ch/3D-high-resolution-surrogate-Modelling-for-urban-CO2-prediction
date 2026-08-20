#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from two_stage_surrogate.data.alignment import linear_time_match, max_time_gap_seconds  # noqa: E402
from two_stage_surrogate.data.sample_manifest import (  # noqa: E402
    Stage1SampleRecord,
    discover_camden_jobs,
    _time_axis_seconds,
)


JOB_RE = re.compile(r"^z(?P<zone>\d+)_camden(?P<year>\d{4})(?P<month>\d{2})$")


# Parse one job entry from the manifest.
def parse_job(job: str) -> tuple[int, int, int]:
    match = JOB_RE.match(job)
    if not match:
        raise ValueError(f"Unexpected Camden job name: {job}")
    return int(match.group("zone")), int(match.group("year")), int(match.group("month"))


# Entry point for the command-line workflow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert V33/V40 coordinate metadata into a Stage 1-style manifest for Stage 2.")
    data_root = Path(os.environ.get("TWOSTAGE_V6_DATA_ROOT", PROJECT_ROOT / "external_data"))
    parser.add_argument("--metadata-root", type=Path, default=data_root / "v40_coordinate_sidecar")
    parser.add_argument("--jobs-root", type=Path, default=Path(os.environ.get("CAMDEN_JOBS_ROOT", data_root / "camden" / "JOBS")))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "generated" / "stage2_manifest")
    parser.add_argument("--patch-d", type=int, default=16)
    parser.add_argument("--patch-h", type=int, default=256)
    parser.add_argument("--patch-w", type=int, default=256)
    args = parser.parse_args()

    meta_manifest = json.loads((args.metadata_root / "metadata_manifest.json").read_text(encoding="utf-8"))
    jobs_all = discover_camden_jobs(args.jobs_root, include_months=list(range(1, 11)))
    job_files = {job.job: job for job in jobs_all}
    time_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    records: list[dict] = []
    preview_rows: list[dict] = []
    for split_info in meta_manifest["splits"]:
        split = split_info["split"]
        split_jobs = list(split_info["jobs"])
        data = np.load(split_info["metadata"], allow_pickle=False)
        count = int(split_info["total_patches"])
        for i in range(count):
            job_name = split_jobs[int(data["job_idx"][i])]
            job = job_files[job_name]
            if job_name not in time_cache:
                out_seconds, _ = _time_axis_seconds(job.out3d)
                dyn_seconds, _ = _time_axis_seconds(job.dynamic)
                chem_seconds, _ = _time_axis_seconds(job.chemistry)
                time_cache[job_name] = (out_seconds, dyn_seconds, chem_seconds)
            out_seconds, dyn_seconds, chem_seconds = time_cache[job_name]
            t_idx = int(data["time_index"][i])
            out_sec = float(out_seconds[t_idx])
            dyn = linear_time_match(out_sec, dyn_seconds)
            chem_idx = int(np.abs(chem_seconds - out_sec).argmin())
            zone, year, month = parse_job(job_name)
            z0 = int(data["z0"][i])
            y0 = int(data["y0"][i])
            x0 = int(data["x0"][i])
            sample_key = f"camden_v40meta/{job_name}/t{t_idx:04d}/z{z0:03d}_y{y0:04d}_x{x0:04d}"
            rec = Stage1SampleRecord(
                sample_key=sample_key,
                split=split,
                job=job_name,
                zone=zone,
                year=year,
                month=month,
                output_time_index=t_idx,
                output_seconds=out_sec,
                dynamic_left_index=int(dyn.left_index),
                dynamic_right_index=int(dyn.right_index),
                dynamic_weight_right=float(dyn.weight_right),
                dynamic_nearest_gap_seconds=float(max_time_gap_seconds(out_sec, dyn_seconds)),
                chemistry_nearest_index=chem_idx,
                chemistry_nearest_gap_seconds=float(abs(float(chem_seconds[chem_idx]) - out_sec)),
                z0=z0,
                y0=y0,
                x0=x0,
                dz=args.patch_d,
                dy=args.patch_h,
                dx=args.patch_w,
            )
            rec_dict = asdict(rec)
            records.append(rec_dict)
            if len(preview_rows) < 5000:
                preview_rows.append(rec_dict)

    jobs_used = [asdict(job_files[name]) for name in sorted({record["job"] for record in records})]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "stage1_manifest_from_v33_v40_metadata",
        "jobs_root": str(args.jobs_root),
        "source_metadata_root": str(args.metadata_root),
        "jobs": jobs_used,
        "patch_shape": [args.patch_d, args.patch_h, args.patch_w],
        "patch_stride": None,
        "record_count": len(records),
        "records": records,
        "notes": [
            "Records are converted from the V33/V40 time-balanced, residual-amplitude-stratified coordinate sidecar.",
            "This keeps Stage 2 sampling close to the final V40 Camden dataset configuration.",
        ],
    }
    (out / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if preview_rows:
        with (out / "stage1_manifest_preview.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(preview_rows[0].keys()))
            writer.writeheader()
            writer.writerows(preview_rows)
    print(json.dumps({"out": str(out), "record_count": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
