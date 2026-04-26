"""Consolidate per-code / per-noise / per-model benchmark CSVs into one file.

Directory layout expected:
    results/<code>/<regime>/<noise>/benchmark_<model>.csv

Output columns:
    code, noise, Distance, Error_Rate(p), Error_Type, model,
    Best_ECR(%), Accuracy(%), Inference_Time(ms), Epochs,
    Learning_Rate, Train_Loss, Val_Loss
"""

import argparse
import re
from pathlib import Path

import pandas as pd

BENCHMARK_RE = re.compile(r"^benchmark_(?P<model>.+)\.csv$")


def collect(results_dir: Path, regime: str | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for code_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        code = code_dir.name
        regime_dirs = (
            [code_dir / regime] if regime else [p for p in code_dir.iterdir() if p.is_dir()]
        )

        for regime_dir in regime_dirs:
            if not regime_dir.is_dir():
                continue

            for noise_dir in sorted(p for p in regime_dir.iterdir() if p.is_dir()):
                noise = noise_dir.name

                for csv_path in sorted(noise_dir.glob("benchmark_*.csv")):
                    match = BENCHMARK_RE.match(csv_path.name)
                    if not match:
                        continue
                    model = match.group("model")

                    df = pd.read_csv(csv_path)
                    if df.empty:
                        continue

                    df.insert(0, "code", code)
                    df.insert(1, "noise", noise)
                    error_type_pos = df.columns.get_loc("Error_Type")
                    df.insert(error_type_pos + 1, "model", model)
                    frames.append(df)

    if not frames:
        raise RuntimeError(f"No benchmark_*.csv found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    default_results = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help=f"Root results directory (default: {default_results})",
    )
    parser.add_argument(
        "--regime",
        default="realistic",
        help="Regime subfolder to include; pass empty string to include all (default: realistic)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-dir>/gathered_stim.csv)",
    )
    args = parser.parse_args()

    regime = args.regime if args.regime else None
    output = args.output or (args.results_dir / "gathered_stim.csv")

    merged = collect(args.results_dir, regime)
    merged.to_csv(output, index=False)
    print(f"Wrote {len(merged):,} rows from {merged['code'].nunique()} codes, "
          f"{merged['noise'].nunique()} noise settings, {merged['model'].nunique()} models -> {output}")


if __name__ == "__main__":
    main()
