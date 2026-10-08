# 30-Day Hospital Readmission Risk (ML + AWS)

**Question:** when a diabetic patient leaves hospital, how likely are they to be back
within 30 days? A care team can only call so many people, so the model's job is to put
the right patients at the top of the call list.

**Data:** real. *Diabetes 130-US Hospitals 1999–2008* (UCI ML Repository, Strack et al.
2014): 101,766 hospital stays, 71,518 patients, 50 columns. `make data` downloads it.

**Stack:** SQL (DuckDB locally, Athena on AWS) · pandas · scikit-learn · LightGBM ·
MLflow · SHAP · ONNX · AWS (S3, Athena, Lambda, IAM, CloudWatch, Budgets) via boto3 ·
pytest + moto · GitHub Actions

```
raw CSV ─► sql/features.sql ─► patient-level split ─► models ─► MLflow (every run)
   │        (same SQL on                                │
   │         DuckDB + Athena)                           ├─► SHAP: what drives risk
   │                                                    ├─► fairness audit by race / sex / age
   ▼                                                    ▼
  S3 ──► Athena                        LightGBM ─► ONNX ─► AWS Lambda (arm64) ─► CloudWatch logs
```

## Results (test split: 14,859 stays from patients the model never saw)

| Model | ROC-AUC | PR-AUC | Readmissions caught when calling the top 20% | Calls that reach a future readmission |
|---|---|---|---|---|
| Rule: number of past inpatient stays | 0.604 | 0.168 | — (ties force a 33% call list) | 17.0% |
| Logistic regression | 0.652 | 0.199 | 36.2% | 20.8% |
| **LightGBM** | **0.665** | **0.216** | **38.6%** | **22.0%** |

Base rate is 11.4%, so a call from the LightGBM list is about **1.9× more likely** to reach
a patient who will be readmitted than a random call. The threshold is fixed on the
validation split (top 20%), not tuned on test. Calibration plot: `artifacts/calibration.png`.

**How good is 0.665?** Honestly, moderate. Published work on this dataset sits in the
mid-0.6s. These are billing-style records: no lab trends, no social factors, no notes.
The ceiling is the data, not the algorithm, and LightGBM's gain over logistic regression
(+0.013 AUC) is small. That's why the README leads with the call-list numbers, not AUC.

## Finding 1: the "wrong" split did not inflate the score here

16,341 patients have more than one stay. The textbook warning: split by stay and the
same patient lands in train and test, so the score is flattered. I tested it instead of
assuming it, with the same model, 5 seeds each (`src/split_experiment.py`):

| Split | Test AUC (mean ± sd, 5 seeds) | Test patients also in train |
|---|---|---|
| By patient (correct) | 0.673 ± 0.005 | 0 |
| By stay (leaky) | 0.673 ± 0.007 | about 4,900 |

**No measurable inflation.** The features describe the stay, not the person (there is no
patient ID in the model), so seeing someone's other stay gives little to memorise.
The project still splits by patient, because that's how the model would meet new
patients in real use. I just don't claim a leakage effect the data doesn't show.

## Finding 2: what drives risk (SHAP, `artifacts/shap_importance.png`)

1. **Past inpatient stays** (by far the strongest)
2. **Discharged home** (lowers risk; transfers to other inpatient care raise it)
3. Age, number of diagnoses, length of stay, primary diagnosis (circulatory, respiratory)

All are clinically sensible. Nothing points at a leaked or post-discharge field.

## Finding 3: fairness audit (`artifacts/fairness.csv`)

Race is **not** a model input. It's kept only to check the model works for everyone.

| Group | Stays | Actual rate | Predicted | Recall at top 20% | AUC |
|---|---|---|---|---|---|
| African American | 2,818 | 11.5% | 11.2% | 41.8% | 0.660 |
| Caucasian | 11,066 | 11.5% | 11.3% | 37.3% | 0.661 |
| Female | 8,094 | 11.4% | 11.5% | 39.3% | 0.664 |
| Male | 6,765 | 11.4% | 10.9% | 37.8% | 0.666 |
| Age < 50 | 2,515 | 11.2% | 10.9% | 51.6% | 0.745 |
| Age 50–69 | 5,718 | 10.1% | 10.4% | 34.4% | 0.635 |
| Age 70–79 | 3,732 | 12.4% | 11.8% | 38.3% | 0.649 |
| Age 80+ | 2,894 | 12.7% | 12.4% | 35.6% | 0.638 |

- **Race and sex:** no meaningful gap. Predictions are calibrated within 0.5 points, and AUC is within 0.01.
- **Age:** the clearest gap. The model ranks **under-50s much better** (AUC 0.745) than
  patients 50 and over (0.635–0.649). Older patients' readmissions are harder to predict
  from these fields.
- Asian, Hispanic and "Other" groups have under 300 test stays each, too few to judge.
  They're reported in the CSV but marked `enough_data = False`.

## Built to run on AWS for $0

| Piece | What it does | Why it costs nothing |
|---|---|---|
| **Budgets** | $1/month alarm, emails on actual or forecast spend | created first, free |
| **S3** | private bucket: raw CSV + Lambda code | a few MB, all public access blocked |
| **Athena** | runs the **same** `sql/features.sql` over S3 and checks totals match DuckDB | scans about 19 MB per query, a fraction of a cent |
| **Lambda** | the API: raw stay record in, risk + follow-up flag out (ONNX, arm64, 512 MB) | pay per call; 1M calls/month free |
| **CloudWatch** | one JSON log line per prediction, 7-day retention | tiny volume |

Not used on purpose: EC2, SageMaker endpoints, NAT gateways, public URLs. Nothing runs
or bills while idle. `make destroy` deletes everything and then lists anything still
tagged `project=readmission-risk`, so you can see it's clean.

**Why ONNX instead of Docker:** LightGBM + scikit-learn + scipy are too big for a Lambda
zip. Converted to ONNX, the model runs on `onnxruntime` + `numpy` alone (134 MB unzipped,
limit 250 MB), and the bundle is built from Linux arm64 wheels without Docker. ONNX and
LightGBM agree to 1.6e-7.

## Making sure training and serving agree

The model trains on SQL features but the API can't run SQL per request, so
`src/features.py` re-implements the same rules in Python. Two copies of logic drift apart,
so there are tests for it:

| Test | What it proves |
|---|---|
| `test_feature_parity` | SQL and Python features are identical on **all 99,340 stays**. Checked that it fails if one rule is broken on purpose (4,922 rows flagged). |
| `test_handler` (6 tests) | the Lambda handler matches LightGBM on 300 test patients; rejects missing fields, hospice patients and bad JSON with a 400; survives unseen categories |
| `test_aws_moto` (2 tests) | deploy then destroy on a fake AWS leaves nothing behind; bucket is private; budget is $1; logs keep 7 days |
| `aws/deploy.py` smoke test | on real AWS: invokes the live Lambda on 20 test patients and compares with local scores |

Everything is reproducible: two clean runs give byte-identical metrics. (They didn't at
first. DuckDB reads the CSV in parallel, so row order changed the "random" split.
Sorting before the shuffle fixed it.)

## Run it

```bash
make setup          # Python 3.11 venv + pinned packages
make data           # download UCI data
make all            # prepare -> train -> split check -> explain -> ONNX + Lambda bundle -> tests
make mlflow         # browse every run at http://127.0.0.1:5000

# AWS (needs `aws configure` with your own keys)
make deploy EMAIL=you@example.com
make destroy
```

## Limits

- **1999–2008 data from 130 US hospitals.** Patterns, coding and care have changed since.
  This shows the method, not a model to put in front of patients.
- **AUC 0.665 is moderate.** Fine for ranking a call list, not for decisions about one person.
- Readmissions to a *different* hospital may be missing from the data, so the true rate
  is probably higher than 11.4%.
- `prior_inpatient` counts visits in the year before the stay. It's legitimate (known at
  discharge), but it means first-time patients are much harder to score.
- The Lambda has no public URL. Calls need AWS credentials. Adding a public API would
  need auth and rate limits first.

## Files

```
sql/features.sql          one row per stay -> features + label (runs on DuckDB and Athena)
src/prepare.py            SQL -> features, patient-level 70/15/15 split, data_report.json
src/features.py           the same feature rules in Python, for the API
src/encoding.py           one-hot schema learned on train only (artifacts/schema.json)
src/train.py              rule / logistic / LightGBM / wrong-split runs -> MLflow
src/split_experiment.py   5-seed by-patient vs by-stay comparison
src/explain.py            SHAP + fairness audit
src/export_onnx.py        ONNX export + Lambda bundle
lambda/handler.py         the API
aws/deploy.py, destroy.py boto3 deploy (6 steps) and full teardown
tests/                    9 tests: feature parity, API, AWS (moto)
```
