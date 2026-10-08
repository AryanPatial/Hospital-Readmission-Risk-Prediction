"""
Step 2: train and compare models, every run tracked in MLflow.

Runs:
  rule_prior_inpatient   no model: rank patients by number of past inpatient stays
  logistic_regression    simple, explainable baseline
  lightgbm               gradient-boosted trees, early-stopped on the validation split
  lightgbm_random_split  same model, but split by STAY instead of by patient, to measure
                         how much the wrong split flatters the score

The care-team framing: a team can call back about 1 in 5 discharged patients. So the
threshold is set on the VALIDATION split to flag the top 20%, and the test split reports
how many real readmissions that catches (recall) and how many calls are useful (precision).
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.encoding import encode_frame, fit_schema, save_schema

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FLAG_SHARE = 0.20
SEED = 42
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=SEED, verbose=-1)


def evaluate(y_valid, p_valid, y_test, p_test, probabilities=True) -> dict:
    threshold = float(np.quantile(p_valid, 1 - FLAG_SHARE))
    flagged = p_test >= threshold
    caught = int((flagged & (y_test == 1)).sum())
    out = {
        "test_roc_auc": round(roc_auc_score(y_test, p_test), 4),
        "test_pr_auc": round(average_precision_score(y_test, p_test), 4),
        "base_rate": round(float(y_test.mean()), 4),
        "threshold": round(threshold, 4),
        "test_flag_rate": round(float(flagged.mean()), 4),
        "test_recall_at_top20": round(caught / int(y_test.sum()), 4),
        "test_precision_at_top20": round(caught / max(int(flagged.sum()), 1), 4),
    }
    if probabilities:
        out["test_brier"] = round(brier_score_loss(y_test, p_test), 4)
    return out


def fit_lgb(X_tr, y_tr, X_va, y_va):
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def main():
    ART.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data/processed/features.parquet")
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))

    schema = fit_schema(tr)
    save_schema(schema, ART / "schema.json")
    X_tr, X_va, X_te = (encode_frame(d, schema) for d in (tr, va, te))
    y_tr, y_va, y_te = (d["readmit_30d"].to_numpy() for d in (tr, va, te))

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("readmission-30d")
    results = {}

    with mlflow.start_run(run_name="rule_prior_inpatient"):
        m = evaluate(y_va, va["prior_inpatient"].to_numpy(float), y_te, te["prior_inpatient"].to_numpy(float),
                     probabilities=False)  # a count is not a probability, so no Brier score
        mlflow.log_metrics(m); results["rule_prior_inpatient"] = m

    with mlflow.start_run(run_name="logistic_regression"):
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        lr.fit(X_tr, y_tr)
        m = evaluate(y_va, lr.predict_proba(X_va)[:, 1], y_te, lr.predict_proba(X_te)[:, 1])
        mlflow.log_params({"C": 0.5, "features": X_tr.shape[1]})
        mlflow.log_metrics(m); results["logistic_regression"] = m

    with mlflow.start_run(run_name="lightgbm"):
        booster = fit_lgb(X_tr, y_tr, X_va, y_va)
        p_te = booster.predict(X_te, num_iteration=booster.best_iteration)
        m = evaluate(y_va, booster.predict(X_va, num_iteration=booster.best_iteration), y_te, p_te)
        mlflow.log_params({**LGB_PARAMS, "best_iteration": booster.best_iteration, "features": X_tr.shape[1]})
        mlflow.log_metrics(m); results["lightgbm"] = m

        booster.save_model(str(ART / "model.txt"), num_iteration=booster.best_iteration)
        (ART / "threshold.json").write_text(json.dumps(
            {"threshold": m["threshold"], "flag_share": FLAG_SHARE}, indent=2))

        frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 0.5], [0, 0.5], "--", color="grey", label="perfect")
        ax.plot(mean_pred, frac_pos, "o-", label="LightGBM")
        ax.set_xlabel("predicted risk"); ax.set_ylabel("actual readmission rate")
        ax.set_title("Calibration on test patients"); ax.legend()
        fig.tight_layout(); fig.savefig(ART / "calibration.png", dpi=120); plt.close(fig)
        mlflow.log_artifact(str(ART / "calibration.png"))
        mlflow.log_artifact(str(ART / "schema.json"))
        mlflow.log_artifact(str(ART / "model.txt"))

    # Same model, wrong split: shuffle STAYS instead of patients, same sizes.
    with mlflow.start_run(run_name="lightgbm_random_split"):
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(df))
        n_tr, n_va = len(tr), len(va)
        X_all, y_all = encode_frame(df, schema).to_numpy(), df["readmit_30d"].to_numpy()
        i_tr, i_va, i_te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
        b2 = fit_lgb(X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va])
        m = evaluate(y_all[i_va], b2.predict(X_all[i_va], num_iteration=b2.best_iteration),
                     y_all[i_te], b2.predict(X_all[i_te], num_iteration=b2.best_iteration))
        test_patients = set(df["patient_nbr"].to_numpy()[i_te])
        train_patients = set(df["patient_nbr"].to_numpy()[i_tr])
        m["test_patients_also_in_train"] = len(test_patients & train_patients)
        mlflow.log_metrics(m); results["lightgbm_random_split"] = m

    (ART / "metrics.json").write_text(json.dumps(results, indent=2))
    print(pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()
