"""
AWS Lambda entry point: one hospital-stay record in, readmission risk out.

Request body (JSON): the raw fields of one stay, same names as the CSV, e.g.
  {"gender": "Female", "age": "[70-80)", "number_inpatient": "2", ...}
Response: {"risk": 0.23, "flag_for_follow_up": true, "threshold": 0.15, "model_version": "..."}

Works for both a direct Lambda invoke (event is the record) and a Function URL /
API Gateway call (record is a JSON string in event["body"]).
Every prediction is logged as one JSON line, which lands in CloudWatch Logs.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.encoding import encode_row, load_schema
from src.features import REQUIRED_FIELDS, build_features, is_eligible

HERE = Path(__file__).resolve().parent
SCHEMA = load_schema(HERE / "schema.json")
THRESHOLD = json.loads((HERE / "threshold.json").read_text())["threshold"]
SESSION = ort.InferenceSession(str(HERE / "model.onnx"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "local")


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def predict(raw: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    raw = {k: str(v) for k, v in raw.items()}
    if not is_eligible(raw):
        raise ValueError("not eligible: patient died, went to hospice, or gender is invalid")
    x = np.asarray([encode_row(build_features(raw), SCHEMA)], dtype=np.float32)
    risk = float(SESSION.run(None, {"input": x})[1][0, 1])
    return {"risk": round(risk, 4), "flag_for_follow_up": risk >= THRESHOLD,
            "threshold": THRESHOLD, "model_version": MODEL_VERSION}


def lambda_handler(event, context=None):
    start = time.perf_counter()
    try:
        raw = json.loads(event["body"]) if isinstance(event, dict) and "body" in event else event
        out = predict(raw)
        status = 200
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        out, status = {"error": str(exc)}, 400
    print(json.dumps({"event": "prediction", "status": status, "risk": out.get("risk"),
                      "flagged": out.get("flag_for_follow_up"),
                      "latency_ms": round((time.perf_counter() - start) * 1000, 2)}))
    return _response(status, out)
