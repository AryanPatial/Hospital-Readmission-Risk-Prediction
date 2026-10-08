"""
Training builds features in SQL; the live API builds them in Python (src/features.py).
If the two ever disagree, the API scores patients on different inputs than the
model was trained on, silently. This runs both over every row and demands an exact match.
"""
import csv
from pathlib import Path

import pandas as pd
import pytest

from src.features import build_features, is_eligible
from src.prepare import RAW, build_features as sql_features

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw data not downloaded (make data)")

COMPARE = [
    "gender", "age_years", "admission_type", "discharge_to", "admission_source",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "prior_outpatient", "prior_emergency", "prior_inpatient", "number_diagnoses",
    "specialty", "primary_diagnosis", "a1c_result", "glucose_result", "insulin",
    "meds_changed", "on_diabetes_meds", "num_dose_changes",
]


def test_sql_and_python_features_match_on_every_row():
    sql = sql_features().set_index("encounter_id")[COMPARE]

    py_rows = {}
    with open(RAW, newline="") as f:
        for raw in csv.DictReader(f):
            if is_eligible(raw):
                py_rows[int(raw["encounter_id"])] = build_features(raw)
    py = pd.DataFrame.from_dict(py_rows, orient="index")[COMPARE]

    assert set(py.index) == set(sql.index), "the two filters keep different rows"
    py = py.loc[sql.index]
    for col in COMPARE:
        a, b = sql[col].astype(str), py[col].astype(str)
        bad = (a != b).sum()
        assert bad == 0, f"{col}: {bad} rows differ, e.g. {a[a != b].head(3).tolist()} vs {b[a != b].head(3).tolist()}"
