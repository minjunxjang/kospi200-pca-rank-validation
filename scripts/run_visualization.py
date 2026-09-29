"""Command-line entry point for the final 3D figure and metric videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import build_visuals as visuals


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render PCA Rank Validation from the three CSV files")
    parser.add_argument("--data", type=Path, default=project / "data" / "processed")
    parser.add_argument("--output", type=Path, default=project / "results" / "generated")
    parser.add_argument("--videos", action="store_true", help="Also render four metric videos")
    args = parser.parse_args()

    visuals.DATA = args.data.resolve()
    visuals.OUT = args.output.resolve()
    visuals.OUT.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ("cases.csv", "full_summary.csv", "annual_summary.csv")
               if not (visuals.DATA / name).is_file()]
    if missing:
        parser.error("Missing CSV files in " + str(visuals.DATA) + ": " + ", ".join(missing))

    _, _, _, years, ks, annual_values, annual_q, _, _, limits = visuals.arrays()
    summary, panels = visuals.make_clear_space_pngs(years, ks, annual_values, annual_q, limits)
    print(f"Combined figure: {summary}")
    for panel in panels:
        print(f"Panel: {panel}")
    if args.videos:
        for video in visuals.make_metric_time_videos(years, ks, annual_values, annual_q, limits):
            print(f"Video: {video}")


if __name__ == "__main__":
    main()
