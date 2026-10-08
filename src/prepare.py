"""
Step 1: raw CSV -> features (via sql/features.sql in DuckDB) -> patient-level split.

Why split by patient: 16,773 patients have more than one hospital stay. A random
split by stay puts the same person in train and test, and the model gets graded
partly on people it has already seen. src/train.py measures how much that inflates
the score.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/diabetic_data.csv"
OUT = ROOT / "data/processed"
SEED = 42


def raw_column_names() -> list[str]:
    """CSV header, lowercased, '-' -> '_'. Athena uses the same list (aws/deploy.py)."""
    with open(RAW) as f:
        return [c.strip().lower().replace("-", "_") for c in f.readline().split(",")]


def build_features() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw_encounters AS SELECT * FROM read_csv('{RAW}', header=true, "
                f"all_varchar=true, names={raw_column_names()})")
    df = con.execute((ROOT / "sql/features.sql").read_text()).df()
    # DuckDB reads the CSV in parallel, so row order can change between runs. Sort it,
    # or the "random" split and every number downstream changes from run to run.
    return df.sort_values("encounter_id").reset_index(drop=True)


def patient_split(df: pd.DataFrame) -> pd.Series:
    """70 / 15 / 15 by patient: every stay of a patient lands in the same split."""
    rng = np.random.default_rng(SEED)
    patients = np.sort(df["patient_nbr"].unique())
    rng.shuffle(patients)
    n = len(patients)
    split_of = {p: "train" for p in patients[: int(0.70 * n)]}
    split_of.update({p: "valid" for p in patients[int(0.70 * n): int(0.85 * n)]})
    split_of.update({p: "test" for p in patients[int(0.85 * n):]})
    return df["patient_nbr"].map(split_of)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows = sum(1 for _ in open(RAW)) - 1
    df = build_features()
    df["split"] = patient_split(df)

    overlap = (df.groupby("patient_nbr")["split"].nunique() > 1).sum()
    assert overlap == 0, f"{overlap} patients in more than one split"

    df.to_parquet(OUT / "features.parquet", index=False)
    report = {
        "raw_stays": raw_rows,
        "stays_after_exclusions": len(df),
        "excluded_died_hospice_or_bad_gender": raw_rows - len(df),
        "patients": int(df["patient_nbr"].nunique()),
        "patients_with_repeat_stays": int((df.groupby("patient_nbr").size() > 1).sum()),
        "readmit_30d_rate": round(float(df["readmit_30d"].mean()), 4),
        "split_sizes": df["split"].value_counts().to_dict(),
        "split_readmit_rate": df.groupby("split")["readmit_30d"].mean().round(4).to_dict(),
        "patients_in_more_than_one_split": int(overlap),
    }
    (OUT / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
