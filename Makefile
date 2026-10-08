PY := .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup data prepare train split explain export test all mlflow deploy destroy clean

setup:            ## create .venv and install everything
	python3.11 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt

data:             ## download the UCI dataset (3 MB zip)
	mkdir -p data/raw
	curl -sL -o data/raw/diabetes.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
	unzip -o -q data/raw/diabetes.zip -d data/raw && rm data/raw/diabetes.zip

prepare:          ## SQL features + patient-level split
	$(PY) -m src.prepare

train:            ## rule, logistic regression, LightGBM (+ wrong-split run) -> MLflow
	$(PY) -m src.train

split:            ## 5-seed check: does splitting by stay inflate the score?
	$(PY) -m src.split_experiment

explain:          ## SHAP + fairness audit
	$(PY) -m src.explain

export:           ## ONNX model + Lambda bundle (Linux arm64 wheels)
	$(PY) -m src.export_onnx --bundle

test:
	$(PY) -m pytest -q tests

all: prepare train split explain export test

mlflow:           ## browse every run at http://127.0.0.1:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

deploy:           ## needs: aws configure, EMAIL=you@example.com
	$(PY) -m aws.deploy --email $(EMAIL)

destroy:          ## delete everything deploy created (keeps the free $1 budget alarm)
	$(PY) -m aws.destroy

clean:
	rm -rf lambda/build data/processed artifacts mlflow.db mlruns
