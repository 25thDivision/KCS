"""Gather all ibm_results_*.csv files in this directory into one combined CSV per backend.

For each backend (ibm_aachen, ibm_pittsburgh, ibm_boston, ibm_miami, ...), the
matching files are concatenated in ascending order of the timestamp embedded in
their filenames, and a new ``Run`` column is inserted (1-indexed) so rows from
the earliest file get Run=1, the next get Run=2, and so on.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FILENAME_RE = re.compile(r"^ibm_results_(\d{8}_\d{6})\.csv$")


def main() -> None:
    files_by_timestamp: list[tuple[str, Path]] = []
    for path in HERE.iterdir():
        m = FILENAME_RE.match(path.name)
        if m:
            files_by_timestamp.append((m.group(1), path))
    files_by_timestamp.sort(key=lambda x: x[0])

    if not files_by_timestamp:
        print(f"No ibm_results_*.csv files found in {HERE}")
        return

    groups: dict[str, list[pd.DataFrame]] = defaultdict(list)
    run_counters: dict[str, int] = defaultdict(int)

    for timestamp, path in files_by_timestamp:
        df = pd.read_csv(path)
        backends = df["Backend"].unique()
        if len(backends) != 1:
            raise ValueError(
                f"{path.name} contains multiple backends {backends!r}; "
                "expected exactly one per file."
            )
        backend = backends[0]
        run_counters[backend] += 1
        backend_idx = df.columns.get_loc("Backend")
        df.insert(backend_idx + 1, "Run", run_counters[backend])
        groups[backend].append(df)
        print(f"  {path.name}  ->  {backend}  (Run={run_counters[backend]})")

    for backend, frames in groups.items():
        combined = pd.concat(frames, ignore_index=True)
        out_path = HERE / f"combined_{backend}.csv"
        combined.to_csv(out_path, index=False)
        print(f"Wrote {out_path.name}: {len(frames)} file(s), {len(combined)} rows")


if __name__ == "__main__":
    main()
