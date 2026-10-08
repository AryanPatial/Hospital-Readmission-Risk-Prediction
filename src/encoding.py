"""
Turn feature dicts into the number matrix the model reads (one-hot for categories).

The category levels are learned from the TRAINING split only and saved to
artifacts/schema.json. The Lambda API loads the same file, so training and serving
build columns in exactly the same order. A category never seen in training
becomes all zeros instead of crashing the API.
"""
from __future__ import annotations
import json

NUMERIC = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "prior_outpatient", "prior_emergency", "prior_inpatient",
    "number_diagnoses", "meds_changed", "on_diabetes_meds", "num_dose_changes",
]
CATEGORICAL = [
    "gender", "admission_type", "discharge_to", "admission_source", "specialty",
    "primary_diagnosis", "a1c_result", "glucose_result", "insulin",
]


def fit_schema(train_rows) -> dict:
    """train_rows: a pandas DataFrame of the training split."""
    levels = {c: sorted(str(v) for v in train_rows[c].unique()) for c in CATEGORICAL}
    columns = list(NUMERIC) + [f"{c}={lvl}" for c in CATEGORICAL for lvl in levels[c]]
    return {"numeric": NUMERIC, "categorical": levels, "columns": columns}


def encode_row(feats: dict, schema: dict) -> list[float]:
    row = [float(feats[c]) for c in schema["numeric"]]
    for c, lvls in schema["categorical"].items():
        val = str(feats[c])
        row.extend(1.0 if val == lvl else 0.0 for lvl in lvls)
    return row


def encode_frame(df, schema: dict):
    """Vectorised version for training. Same column order as encode_row."""
    import pandas as pd
    parts = [df[schema["numeric"]].astype("float32")]
    for c, lvls in schema["categorical"].items():
        vals = df[c].astype(str)
        parts.append(pd.DataFrame({f"{c}={lvl}": (vals == lvl).astype("float32") for lvl in lvls},
                                  index=df.index))
    return pd.concat(parts, axis=1)[schema["columns"]]


def save_schema(schema: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_schema(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
