"""
The live API must give the same risk as the trained model, and reject bad input cleanly.
Runs the real handler (ONNX) on test-split patients and compares with LightGBM.
"""
import csv
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.prepare import RAW, ROOT

ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model.onnx").exists() or not RAW.exists(),
                                reason="run make train export first")


@pytest.fixture(scope="module")
def test_records():
    feats = pd.read_parquet(ROOT / "data/processed/features.parquet")
    test_ids = set(feats.loc[feats.split == "test", "encounter_id"].head(300))
    with open(RAW, newline="") as f:
        return [r for r in csv.DictReader(f) if int(r["encounter_id"]) in test_ids]


def _load_handler():
    """Load lambda/handler.py the way Lambda sees it: model files next to the handler."""
    import importlib.util, shutil, tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "lambda/handler.py", tmp / "handler.py")
    for f in ("model.onnx", "schema.json", "threshold.json"):
        shutil.copy(ART / f, tmp / f)
    spec = importlib.util.spec_from_file_location("handler_under_test", tmp / "handler.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


H = None


def setup_module():
    global H
    H = _load_handler()


def test_api_matches_trained_model(test_records):
    from src.encoding import encode_frame, load_schema
    from src.features import build_features
    schema = load_schema(ART / "schema.json")
    booster = lgb.Booster(model_file=str(ART / "model.txt"))
    expected = booster.predict(encode_frame(pd.DataFrame([build_features(r) for r in test_records]), schema))
    got = np.array([json.loads(H.lambda_handler(r)["body"])["risk"] for r in test_records])
    assert len(got) == 300
    assert np.max(np.abs(got - expected)) < 1e-4   # API rounds to 4 decimals


def test_function_url_style_event(test_records):
    out = H.lambda_handler({"body": json.dumps(test_records[0])})
    assert out["statusCode"] == 200 and "risk" in json.loads(out["body"])


def test_missing_field_is_400(test_records):
    rec = dict(test_records[0]); rec.pop("number_inpatient")
    out = H.lambda_handler(rec)
    assert out["statusCode"] == 400 and "number_inpatient" in json.loads(out["body"])["error"]


def test_hospice_patient_is_rejected(test_records):
    rec = dict(test_records[0]); rec["discharge_disposition_id"] = "13"
    assert H.lambda_handler(rec)["statusCode"] == 400


def test_unseen_category_does_not_crash(test_records):
    rec = dict(test_records[0]); rec["insulin"] = "SomethingNew"
    assert H.lambda_handler(rec)["statusCode"] == 200


def test_bad_json_body_is_400():
    assert H.lambda_handler({"body": "{not json"})["statusCode"] == 400
