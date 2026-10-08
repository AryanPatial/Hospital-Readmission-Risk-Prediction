"""
Step 3: what drives the model (SHAP), and does it work equally well for everyone?

Fairness audit: race is NOT a model input, but a model can still work worse for a
group through other features. For each race / sex / age group on the test split:
how many readmissions it catches at the top-20% threshold (recall), how often it
flags people, and whether predicted risk matches the real rate (calibration gap).
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
import shap
from sklearn.metrics import roc_auc_score

from src.encoding import encode_frame, load_schema

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MIN_GROUP = 300  # smaller groups are reported but flagged as too small to judge


def group_table(te: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for g, d in te.groupby(col):
        pos = d["readmit_30d"].sum()
        rows.append({
            "attribute": col, "group": str(g), "stays": len(d),
            "actual_rate": round(d["readmit_30d"].mean(), 4),
            "mean_predicted": round(d["risk"].mean(), 4),
            "calibration_gap": round(d["risk"].mean() - d["readmit_30d"].mean(), 4),
            "flag_rate": round(d["flagged"].mean(), 4),
            "recall": round((d["flagged"] & (d["readmit_30d"] == 1)).sum() / pos, 4) if pos else None,
            "roc_auc": round(roc_auc_score(d["readmit_30d"], d["risk"]), 4) if 0 < pos < len(d) else None,
            "enough_data": len(d) >= MIN_GROUP,
        })
    return pd.DataFrame(rows)


def main():
    schema = load_schema(ART / "schema.json")
    threshold = json.loads((ART / "threshold.json").read_text())["threshold"]
    booster = lgb.Booster(model_file=str(ART / "model.txt"))

    df = pd.read_parquet(ROOT / "data/processed/features.parquet")
    te = df[df.split == "test"].copy()
    X_te = encode_frame(te, schema)
    te["risk"] = booster.predict(X_te)
    te["flagged"] = te["risk"] >= threshold
    te["age_band"] = pd.cut(te["age_years"], [0, 50, 70, 80, 200], right=False,
                            labels=["<50", "50-69", "70-79", "80+"])

    # SHAP on a fixed sample of test patients
    sample = X_te.sample(3000, random_state=42)
    sv = shap.TreeExplainer(booster).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    top = imp.head(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#1f5fbf")
    ax.set_xlabel("mean |SHAP value| (impact on predicted log-odds)")
    ax.set_title("What drives predicted readmission risk")
    fig.tight_layout(); fig.savefig(ART / "shap_importance.png", dpi=120); plt.close(fig)

    fairness = pd.concat([group_table(te, c) for c in ("race", "gender", "age_band")], ignore_index=True)
    fairness.to_csv(ART / "fairness.csv", index=False)
    (ART / "shap_top_features.json").write_text(json.dumps(imp.head(15).round(4).to_dict(), indent=2))

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("readmission-30d")
    with mlflow.start_run(run_name="explain_and_fairness"):
        for f in ("shap_importance.png", "fairness.csv", "shap_top_features.json"):
            mlflow.log_artifact(str(ART / f))

    print(imp.head(12).round(4).to_string()); print()
    print(fairness.to_string(index=False))


if __name__ == "__main__":
    main()
