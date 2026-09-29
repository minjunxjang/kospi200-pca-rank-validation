from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_pca_rank_validation import K_VALUES, OUT, bh_qvalues, hac_mean_test


ROOT = Path(__file__).resolve().parents[1]
CSV_OUT = ROOT / "results" / "recomputed" / "csv"


def main():
    cases = pd.read_csv(CSV_OUT / "cases.csv")
    cubes = np.load(OUT / "rank_cubes.npz")
    dates = pd.to_datetime(cubes["dates"])
    years = dates.year.to_numpy()
    year_values = np.unique(years)
    metric_defs = [
        ("rank_ic", cubes["daily_rank_ic"], 0.0),
        ("quintile_ic", cubes["daily_quintile_ic"], 0.0),
        ("delta_rank_ic", cubes["daily_delta_rank_ic"], 0.0),
        ("delta_quintile_ic", cubes["daily_delta_quintile_ic"], 0.0),
    ]
    full_rows, annual_rows = [], []
    for metric, cube, null in metric_defs:
        for ci, row in cases.iterrows():
            if metric.startswith("delta_") and bool(row.is_raw):
                continue
            for ki, k in enumerate(K_VALUES):
                lag = max(int(row.n), k) - 1
                full_rows.append({"metric": metric, **row.to_dict(), "k": k, "null": null, **hac_mean_test(cube[:, ci, ki], null, lag)})
                for yr in year_values:
                    annual_rows.append({"metric": metric, **row.to_dict(), "k": k, "year": int(yr), "null": null, **hac_mean_test(cube[years == yr, ci, ki], null, lag)})
    full, annual = pd.DataFrame(full_rows), pd.DataFrame(annual_rows)
    for metric in full.metric.unique():
        mask = full.metric == metric
        full.loc[mask, "q_pos"] = bh_qvalues(full.loc[mask, "p_pos"].to_numpy())
        full.loc[mask, "q_two"] = bh_qvalues(full.loc[mask, "p_two"].to_numpy())
    for metric in annual.metric.unique():
        mask_m = annual.metric == metric
        annual.loc[mask_m, "q_pos_global_space"] = bh_qvalues(annual.loc[mask_m, "p_pos"].to_numpy())
        annual.loc[mask_m, "q_two_global_space"] = bh_qvalues(annual.loc[mask_m, "p_two"].to_numpy())
        for yr in year_values:
            mask = mask_m & (annual.year == yr)
            annual.loc[mask, "q_pos_within_year"] = bh_qvalues(annual.loc[mask, "p_pos"].to_numpy())
            annual.loc[mask, "q_two_within_year"] = bh_qvalues(annual.loc[mask, "p_two"].to_numpy())
    full.to_csv(CSV_OUT / "full_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(CSV_OUT / "annual_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
