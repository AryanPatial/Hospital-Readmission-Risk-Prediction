"""
The same feature rules as sql/features.sql, in plain Python, for the live API.

Training uses the SQL. The Lambda API can't run SQL on one request, so it uses this.
Two copies of the same logic can drift apart (training/serving skew), so
tests/test_feature_parity.py runs both over every row and fails on any difference.

No pandas or numpy here: this file ships inside the Lambda zip.
"""
from __future__ import annotations

EXCLUDED_DISCHARGE = {"11", "13", "14", "19", "20", "21"}  # died or hospice

TOP_SPECIALTIES = {
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
    "Surgery-General", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive",
    "Radiologist", "Pulmonology", "Psychiatry",
}

DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC_INPUTS = {
    "time_in_hospital": "time_in_hospital",
    "num_lab_procedures": "num_lab_procedures",
    "num_procedures": "num_procedures",
    "num_medications": "num_medications",
    "number_outpatient": "prior_outpatient",
    "number_emergency": "prior_emergency",
    "number_inpatient": "prior_inpatient",
    "number_diagnoses": "number_diagnoses",
}

# Raw fields the API needs. Drug columns are optional (missing = "No").
REQUIRED_FIELDS = [
    "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "diag_1", "A1Cresult",
    "max_glu_serum", "change", "diabetesMed", *NUMERIC_INPUTS,
]


def is_eligible(raw: dict) -> bool:
    """Same filter as the SQL WHERE clause."""
    return (str(raw["discharge_disposition_id"]) not in EXCLUDED_DISCHARGE
            and raw["gender"] in ("Male", "Female"))


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _diag_group(code: str) -> str:
    if code == "?" or code.startswith(("V", "E")):
        return "other"
    if code.startswith("250"):
        return "diabetes"
    v = _num(code)
    if v is None:
        return "other"
    if 390 <= v <= 459.99 or 785 <= v <= 785.99:
        return "circulatory"
    if 460 <= v <= 519.99 or 786 <= v <= 786.99:
        return "respiratory"
    if 520 <= v <= 579.99 or 787 <= v <= 787.99:
        return "digestive"
    if 800 <= v <= 999.99:
        return "injury"
    if 710 <= v <= 739.99:
        return "musculoskeletal"
    if 580 <= v <= 629.99 or 788 <= v <= 788.99:
        return "genitourinary"
    if 140 <= v <= 239.99:
        return "neoplasms"
    return "other"


def _lab(value: str) -> str:
    return {"None": "not_tested", "Norm": "normal"}.get(value, "high")


def build_features(raw: dict) -> dict:
    """One raw hospital-stay record (strings, as in the CSV) -> model features."""
    adm_type = str(raw["admission_type_id"])
    disch = str(raw["discharge_disposition_id"])
    source = str(raw["admission_source_id"])
    age = raw["age"]

    if disch == "1":
        discharge_to = "home"
    elif disch == "6":
        discharge_to = "home_health"
    elif disch in ("3", "4", "24"):
        discharge_to = "nursing_facility"
    elif disch in ("2", "5", "22", "23"):
        discharge_to = "other_inpatient"
    elif disch == "7":
        discharge_to = "left_ama"
    else:
        discharge_to = "other"

    if source == "7":
        admission_source = "emergency"
    elif source in ("1", "2", "3"):
        admission_source = "referral"
    elif source in ("4", "5", "6", "10", "22", "25"):
        admission_source = "transfer"
    else:
        admission_source = "other"

    spec = raw["medical_specialty"]
    feats = {
        "gender": raw["gender"],
        "age_years": int(age[1:age.index("-")]) + 5,
        "admission_type": adm_type if adm_type in ("1", "2", "3", "7") else "other",
        "discharge_to": discharge_to,
        "admission_source": admission_source,
        "specialty": "Unknown" if spec == "?" else (spec if spec in TOP_SPECIALTIES else "Other"),
        "primary_diagnosis": _diag_group(str(raw["diag_1"])),
        "a1c_result": _lab(raw["A1Cresult"]),
        "glucose_result": _lab(raw["max_glu_serum"]),
        "insulin": raw.get("insulin", "No"),
        "meds_changed": 1 if raw["change"] == "Ch" else 0,
        "on_diabetes_meds": 1 if raw["diabetesMed"] == "Yes" else 0,
        "num_dose_changes": sum(1 for c in DRUG_COLUMNS if raw.get(c, "No") in ("Up", "Down")),
    }
    for src, dst in NUMERIC_INPUTS.items():
        feats[dst] = int(raw[src])
    return feats
