"""
Step 4: convert the LightGBM model to ONNX for the Lambda API.

Why ONNX: LightGBM + scikit-learn + scipy are too big for a Lambda zip. The ONNX
runtime + numpy fit easily, and no Docker is needed. Then the Lambda bundle
(lambda/build/) is assembled with Linux arm64 wheels.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from src.encoding import load_schema

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
BUILD = ROOT / "lambda/build"


def export():
    schema = load_schema(ART / "schema.json")
    booster = lgb.Booster(model_file=str(ART / "model.txt"))
    onx = convert_lightgbm(booster, initial_types=[("input", FloatTensorType([None, len(schema["columns"])]))],
                           target_opset=15, zipmap=False)
    (ART / "model.onnx").write_bytes(onx.SerializeToString())

    # check: ONNX and LightGBM give the same risk on random rows
    rng = np.random.default_rng(0)
    X = rng.random((500, len(schema["columns"])), dtype=np.float32) * 10
    sess = ort.InferenceSession(str(ART / "model.onnx"))
    p_onnx = sess.run(None, {"input": X})[1][:, 1]
    gap = float(np.max(np.abs(p_onnx - booster.predict(X))))
    print(f"ONNX vs LightGBM max difference: {gap:.2e}")
    assert gap < 1e-5


def bundle():
    """lambda/build = handler + feature code + model files + Linux arm64 wheels."""
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True)
    # Lambda python3.12 runs Amazon Linux 2023 (glibc 2.34), so manylinux_2_28 wheels work there.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(BUILD),
                    "--platform", "manylinux2014_aarch64", "--platform", "manylinux_2_28_aarch64",
                    "--implementation", "cp", "--python-version", "3.12", "--only-binary=:all:",
                    "numpy<2.3", "onnxruntime==1.20.1"], check=True)
    for junk in BUILD.glob("**/tests"):
        shutil.rmtree(junk, ignore_errors=True)
    shutil.copy(ROOT / "lambda/handler.py", BUILD / "handler.py")
    (BUILD / "src").mkdir()
    for f in ("__init__.py", "features.py", "encoding.py"):
        shutil.copy(ROOT / "src" / f, BUILD / "src" / f)
    for f in ("model.onnx", "schema.json", "threshold.json"):
        shutil.copy(ART / f, BUILD / f)
    size = sum(p.stat().st_size for p in BUILD.rglob("*") if p.is_file()) / 1e6
    print(f"Lambda bundle: {size:.0f} MB unzipped (limit 250 MB)")
    assert size < 250


if __name__ == "__main__":
    export()
    if "--bundle" in sys.argv:
        bundle()
