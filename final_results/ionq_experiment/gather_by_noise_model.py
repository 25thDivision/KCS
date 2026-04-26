"""Gather all ionq_results_*.csv files in this directory into one combined CSV per noise model.

For each noise model (forte-1, ...), files are grouped by their date prefix and
ordered by timestamp within each date. A ``Run`` column is inserted right after
``Noise_Model``: the Nth file (within its date) shares Run=N across dates, so
rows from different distances captured in the same experimental run are paired.
``DATE_DISTANCE_FILTER`` selects which Distance to keep per date.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FILENAME_RE = re.compile(r"^ionq_results_(\d{8}_\d{6})\.csv$")

# Per-date filter: only rows whose Distance matches the listed value are kept.
# Dates not listed here keep all rows.
DATE_DISTANCE_FILTER: dict[str, int] = {
    "20260415": 3,
    "20260426": 5,
}


def main() -> None:
    files_by_timestamp: list[tuple[str, Path]] = []
    for path in HERE.iterdir():
        m = FILENAME_RE.match(path.name)
        if m:
            files_by_timestamp.append((m.group(1), path))
    files_by_timestamp.sort(key=lambda x: x[0])

    if not files_by_timestamp:
        print(f"No ionq_results_*.csv files found in {HERE}")
        return

    # Bucket files by date prefix, preserving timestamp order within each date.
    files_by_date: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for timestamp, path in files_by_timestamp:
        date_prefix = timestamp.split("_", 1)[0]
        files_by_date[date_prefix].append((timestamp, path))

    groups: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for date_prefix, entries in files_by_date.items():
        keep_distance = DATE_DISTANCE_FILTER.get(date_prefix)
        for run_idx, (timestamp, path) in enumerate(entries, start=1):
            df = pd.read_csv(path)
            if keep_distance is not None:
                df = df[df["Distance"] == keep_distance]
                if df.empty:
                    print(f"  {path.name}  ->  skipped (no rows with Distance={keep_distance})")
                    continue
            noise_models = df["Noise_Model"].unique()
            if len(noise_models) != 1:
                raise ValueError(
                    f"{path.name} contains multiple noise models {noise_models!r}; "
                    "expected exactly one per file."
                )
            noise_model = noise_models[0]
            noise_idx = df.columns.get_loc("Noise_Model")
            df.insert(noise_idx + 1, "Run", run_idx)
            groups[noise_model].append(df)
            filter_note = f", Distance={keep_distance}" if keep_distance is not None else ""
            print(f"  {path.name}  ->  {noise_model}  (Run={run_idx}{filter_note})")

    for noise_model, frames in groups.items():
        combined = pd.concat(frames, ignore_index=True)
        out_path = HERE / f"combined_{noise_model}.csv"
        combined.to_csv(out_path, index=False)
        print(f"Wrote {out_path.name}: {len(frames)} file(s), {len(combined)} rows")


if __name__ == "__main__":
    main()
