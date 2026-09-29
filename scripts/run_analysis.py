"""Command-line entry point for the PCA rank validation calculation."""

from __future__ import annotations

import argparse
from pathlib import Path


SOURCE_FILES = (
    "00_COMMON252_BASE.xlsx",
    "02_PCA_W60_CUMULATIVE.xlsx",
    "04_PCA_W120_CUMULATIVE.xlsx",
    "06_PCA_W252_CUMULATIVE.xlsx",
    "FUTURE_LOG_RETURNS_5_10_20_60.xlsx",
)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calculate PCA Rank Validation from five XLSX workbooks")
    parser.add_argument("--source-xlsx-dir", type=Path, default=project / "data" / "raw")
    parser.add_argument("--output", type=Path, default=project / "results" / "recomputed")
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Recalculate full and annual summaries from an existing rank_cubes.npz",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    intermediate = output / "intermediate"
    csv_output = output / "csv"

    if args.summaries_only:
        if not (intermediate / "rank_cubes.npz").is_file():
            parser.error(f"Missing {intermediate / 'rank_cubes.npz'}")
        if not (csv_output / "cases.csv").is_file():
            parser.error(f"Missing {csv_output / 'cases.csv'}")
    else:
        source = args.source_xlsx_dir.resolve()
        missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
        if missing:
            parser.error("Missing input workbooks in " + str(source) + ": " + ", ".join(missing))

    import run_pca_rank_validation as analysis

    analysis.OUT = intermediate
    analysis.CSV_OUT = csv_output
    analysis.OUT.mkdir(parents=True, exist_ok=True)
    analysis.CSV_OUT.mkdir(parents=True, exist_ok=True)

    if args.summaries_only:
        import recompute_summaries

        recompute_summaries.OUT = analysis.OUT
        recompute_summaries.CSV_OUT = analysis.CSV_OUT
        recompute_summaries.main()
    else:
        analysis.ROOT = source
        analysis.main()

    print(f"CSV output: {analysis.CSV_OUT}")


if __name__ == "__main__":
    main()
