from __future__ import annotations

import csv
import json
from pathlib import Path


# Read a JSON metadata or configuration file.
def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Write a JSON metadata or configuration file.
def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Write epoch metrics to a CSV history file.
def save_history(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Load previous epoch metrics when resuming training.
def load_history(path: str | Path) -> list[dict[str, float]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = float("nan")
                    continue
                if key == "epoch":
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows
