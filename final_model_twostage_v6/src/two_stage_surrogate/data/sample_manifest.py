from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from .alignment import linear_time_match, max_time_gap_seconds, values_to_seconds
from .patching import PatchSlice, iter_patch_slices


JOB_RE = re.compile(r"^z(?P<zone>\d+)_camden(?P<year>\d{4})(?P<month>\d{2})$")


# Stores PALM file paths for one Stage 1 simulation job.
@dataclass(frozen=True)
class Stage1JobFiles:
    job: str
    zone: int
    year: int
    month: int
    job_dir: str
    dynamic: str
    static: str
    chemistry: str
    out3d: str


# Stores metadata for one Stage 1 training sample.
@dataclass(frozen=True)
class Stage1SampleRecord:
    sample_key: str
    split: str
    job: str
    zone: int
    year: int
    month: int
    output_time_index: int
    output_seconds: float
    dynamic_left_index: int
    dynamic_right_index: int
    dynamic_weight_right: float
    dynamic_nearest_gap_seconds: float
    chemistry_nearest_index: int
    chemistry_nearest_gap_seconds: float
    z0: int
    y0: int
    x0: int
    dz: int
    dy: int
    dx: int


# Discover Camden simulation jobs for Stage 1.
def discover_camden_jobs(jobs_root: str | Path, include_months: list[int] | None = None) -> list[Stage1JobFiles]:
    root = Path(jobs_root)
    include = set(include_months) if include_months else None
    jobs: list[Stage1JobFiles] = []
    for job_dir in sorted(root.iterdir()):
        if not job_dir.is_dir():
            continue
        match = JOB_RE.match(job_dir.name)
        if not match:
            continue
        month = int(match.group("month"))
        if include is not None and month not in include:
            continue
        job = job_dir.name
        files = Stage1JobFiles(
            job=job,
            zone=int(match.group("zone")),
            year=int(match.group("year")),
            month=month,
            job_dir=str(job_dir),
            dynamic=str(job_dir / "INPUT" / f"{job}_dynamic"),
            static=str(job_dir / "INPUT" / f"{job}_static"),
            chemistry=str(job_dir / "INPUT" / f"{job}_chemistry"),
            out3d=str(job_dir / "OUTPUT" / f"{job}_3d_N02.000.nc"),
        )
        if all(Path(p).exists() for p in (files.dynamic, files.static, files.chemistry, files.out3d)):
            jobs.append(files)
    return jobs


# Internal helper for time axis seconds.
def _time_axis_seconds(dataset_path: str, coord_name: str = "time") -> tuple[np.ndarray, str]:
    with Dataset(dataset_path, "r") as ds:
        if coord_name not in ds.variables:
            raise KeyError(f"{dataset_path} does not contain coordinate {coord_name!r}")
        coord = ds.variables[coord_name]
        units = str(getattr(coord, "units", "seconds"))
        return values_to_seconds(np.asarray(coord[:]), units), units


# Internal helper for output grid shape.
def _output_grid_shape(out3d_path: str) -> tuple[int, int, int]:
    with Dataset(out3d_path, "r") as ds:
        required = ["zu_3d", "y", "x"]
        missing = [name for name in required if name not in ds.dimensions]
        if missing:
            raise KeyError(f"{out3d_path} missing output dimensions {missing}")
        return tuple(int(len(ds.dimensions[name])) for name in required)


# Internal helper for split for time index.
def _split_for_time_index(time_index: int, train_fraction: float, val_fraction: float, total: int) -> str:
    train_end = int(round(total * train_fraction))
    val_end = int(round(total * (train_fraction + val_fraction)))
    if time_index < train_end:
        return "train"
    if time_index < val_end:
        return "val"
    return "dev"


# Internal helper for records for job.
def _records_for_job(
    job: Stage1JobFiles,
    patch_slices: list[PatchSlice],
    train_fraction: float,
    val_fraction: float,
) -> list[Stage1SampleRecord]:
    output_seconds, _ = _time_axis_seconds(job.out3d)
    dynamic_seconds, _ = _time_axis_seconds(job.dynamic)
    chemistry_seconds, _ = _time_axis_seconds(job.chemistry)
    records: list[Stage1SampleRecord] = []
    for t_idx, out_sec in enumerate(output_seconds):
        dyn = linear_time_match(float(out_sec), dynamic_seconds)
        chem_idx = int(np.abs(chemistry_seconds - float(out_sec)).argmin())
        split = _split_for_time_index(t_idx, train_fraction, val_fraction, len(output_seconds))
        for patch in patch_slices:
            key = f"camden/{job.job}/t{t_idx:04d}/z{patch.z0:03d}_y{patch.y0:04d}_x{patch.x0:04d}"
            records.append(
                Stage1SampleRecord(
                    sample_key=key,
                    split=split,
                    job=job.job,
                    zone=job.zone,
                    year=job.year,
                    month=job.month,
                    output_time_index=int(t_idx),
                    output_seconds=float(out_sec),
                    dynamic_left_index=dyn.left_index,
                    dynamic_right_index=dyn.right_index,
                    dynamic_weight_right=dyn.weight_right,
                    dynamic_nearest_gap_seconds=max_time_gap_seconds(float(out_sec), dynamic_seconds),
                    chemistry_nearest_index=chem_idx,
                    chemistry_nearest_gap_seconds=abs(float(chemistry_seconds[chem_idx]) - float(out_sec)),
                    z0=patch.z0,
                    y0=patch.y0,
                    x0=patch.x0,
                    dz=patch.dz,
                    dy=patch.dy,
                    dx=patch.dx,
                )
            )
    return records


# Build stage1 manifest for the workflow.
def build_stage1_manifest(config: dict[str, Any]) -> dict[str, Any]:
    jobs = discover_camden_jobs(config["jobs_root"], include_months=config.get("include_months"))
    if not jobs:
        raise RuntimeError("No Camden jobs found for Stage 1 manifest")
    if config.get("limit_jobs"):
        jobs = jobs[: int(config["limit_jobs"])]

    nz, ny, nx = _output_grid_shape(jobs[0].out3d)
    patch_shape = tuple(int(v) for v in config["patch_shape"])
    strides = tuple(int(v) for v in config["patch_stride"])
    patch_slices = iter_patch_slices((nz, ny, nx), patch_shape, strides)
    if config.get("z_start_values") is not None:
        allowed = {int(v) for v in config["z_start_values"]}
        patch_slices = [p for p in patch_slices if p.z0 in allowed]

    records: list[Stage1SampleRecord] = []
    for job in jobs:
        records.extend(
            _records_for_job(
                job,
                patch_slices,
                float(config.get("train_fraction", 0.70)),
                float(config.get("val_fraction", 0.15)),
            )
        )

    return {
        "schema": "stage1_manifest_v1",
        "jobs_root": str(config["jobs_root"]),
        "jobs": [asdict(job) for job in jobs],
        "patch_shape": list(patch_shape),
        "patch_stride": list(strides),
        "record_count": len(records),
        "records": [asdict(record) for record in records],
        "notes": [
            "Manifest records metadata only; no high-resolution arrays are cached here.",
            "Dynamic forcing alignment records linear left/right indices and weights.",
            "Chemistry alignment records nearest hourly emission index.",
        ],
    }


# Write manifest to disk.
def write_manifest(manifest: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "stage1_manifest.json"
    csv_path = out / "stage1_manifest_preview.csv"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    records = manifest["records"]
    if records:
        import csv

        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records[: min(5000, len(records))])
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path

