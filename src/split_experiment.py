"""
Does splitting by stay (wrong) instead of by patient (right) inflate the score here?

One train/test split is noisy, so both methods are repeated with 5 seeds, same model,
same split sizes. Logged to MLflow; summary in artifacts/split_experiment.json.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.encoding import encode_frame, load_schema
from src.train import fit_lgb

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 2, 3, 4, 5]


def split_indices(df, seed, by_patient):
    rng = np.random.default_rng(seed)
    if by_patient:
        pats = df["patient_nbr"].unique(); rng.shuffle(pats)
        n = len(pats)
        tr = set(pats[: int(.70 * n)]); va = set(pats[int(.70 * n): int(.85 * n)])
        lab = df["patient_nbr"].map(lambda p: 0 if p in tr else (1 if p in va else 2)).to_numpy()
    else:
        order = rng.permutation(len(df)); n = len(df)
        lab = np.empty(n, int)
        lab[order[: int(.70 * n)]] = 0; lab[order[int(.70 * n): int(.85 * n)]] = 1; lab[order[int(.85 * n):]] = 2
    return (np.where(lab == k)[0] for k in (0, 1, 2))


def main():
    df = pd.read_parquet(ROOT / "data/processed/features.parquet").reset_index(drop=True)
    X = encode_frame(df, load_schema(ROOT / "artifacts/schema.json")).to_numpy()
    y = df["readmit_30d"].to_numpy()
    pid = df["patient_nbr"].to_numpy()

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("readmission-split-experiment")
    rows = []
    for by_patient in (True, False):
        for seed in SEEDS:
            i_tr, i_va, i_te = split_indices(df, seed, by_patient)
            model = fit_lgb(X[i_tr], y[i_tr], X[i_va], y[i_va])
            auc = roc_auc_score(y[i_te], model.predict(X[i_te], num_iteration=model.best_iteration))
            shared = len(set(pid[i_te]) & set(pid[i_tr]))
            name = "by_patient" if by_patient else "by_stay"
            with mlflow.start_run(run_name=f"{name}_seed{seed}"):
                mlflow.log_params({"split": name, "seed": seed})
                mlflow.log_metrics({"test_roc_auc": auc, "test_patients_also_in_train": shared})
            rows.append({"split": name, "seed": seed, "auc": auc, "shared_patients": shared})

    res = pd.DataFrame(rows)
    summary = res.groupby("split").agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                                       shared_patients_mean=("shared_patients", "mean")).round(4)
    print(res.round(4).to_string(index=False)); print(); print(summary.to_string())
    (ROOT / "artifacts/split_experiment.json").write_text(json.dumps(
        {"runs": res.round(4).to_dict(orient="records"), "summary": summary.reset_index().to_dict(orient="records")},
        indent=2))


if __name__ == "__main__":
    main()
