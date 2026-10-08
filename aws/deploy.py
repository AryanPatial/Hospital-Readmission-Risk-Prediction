"""
Deploy to AWS with boto3. Every step is safe to re-run. Everything created is tagged
project=readmission-risk and recorded in aws/state.json so aws/destroy.py can remove it.

  python -m aws.deploy --email you@example.com

Steps:
  1. $1 monthly budget that emails you on any spend (first, before anything else)
  2. private S3 bucket: raw CSV + Lambda code
  3. Athena: table over the raw CSV, then run sql/features.sql there and check it
     gives the same totals as DuckDB locally
  4. IAM role + Lambda function (arm64, Python 3.12) serving the ONNX model
  5. CloudWatch log group with 7-day retention
  6. smoke test: invoke Lambda on real test patients, compare with local scores

What is deliberately NOT created: EC2, SageMaker endpoints, NAT gateways, public
URLs. Nothing here runs or bills while idle.
"""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "aws/state.json"
PROJECT = "readmission-risk"
TAGS = {"project": PROJECT}
FUNCTION = "readmission-risk-api"
ROLE = "readmission-risk-lambda-role"
LOG_GROUP = f"/aws/lambda/{FUNCTION}"
DATABASE = "readmission_risk"


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


# ---------- 1. budget ----------
def ensure_budget(session, account: str, email: str) -> None:
    budgets = session.client("budgets", region_name="us-east-1")
    name = f"{PROJECT}-1-dollar"
    existing = [b["BudgetName"] for b in budgets.describe_budgets(AccountId=account).get("Budgets", [])]
    if name in existing:
        return
    budgets.create_budget(
        AccountId=account,
        Budget={"BudgetName": name, "BudgetLimit": {"Amount": "1", "Unit": "USD"},
                "TimeUnit": "MONTHLY", "BudgetType": "COST"},
        NotificationsWithSubscribers=[
            {"Notification": {"NotificationType": kind, "ComparisonOperator": "GREATER_THAN",
                              "Threshold": 1.0, "ThresholdType": "PERCENTAGE"},
             "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}]}
            for kind in ("ACTUAL", "FORECASTED")])


# ---------- 2. S3 ----------
def ensure_bucket(s3, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except s3.exceptions.ClientError:
        pass
    kwargs = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
    s3.create_bucket(Bucket=bucket, **kwargs)
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in TAGS.items()]})


def zip_lambda() -> bytes:
    build = ROOT / "lambda/build"
    if not (build / "handler.py").exists():
        raise SystemExit("lambda/build is empty: run `make export` first")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in build.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(build))
    return buf.getvalue()


# ---------- 3. Athena ----------
def athena_query(athena, sql: str, output: str) -> list[list[str]]:
    qid = athena.start_query_execution(QueryString=sql, ResultConfiguration={"OutputLocation": output})["QueryExecutionId"]
    while True:
        st = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if st["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)
    if st["State"] != "SUCCEEDED":
        raise RuntimeError(f"Athena query {st['State']}: {st.get('StateChangeReason')}")
    rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    return [[c.get("VarCharValue") for c in r["Data"]] for r in rows]


CHECK_SQL = """SELECT count(*) AS stays, sum(readmit_30d) AS readmits, sum(age_years) AS age_sum,
       sum(num_dose_changes) AS dose_changes, count(DISTINCT primary_diagnosis) AS diag_groups
FROM ({features}) f"""


def setup_athena(athena, bucket: str) -> dict:
    from src.prepare import raw_column_names
    out = f"s3://{bucket}/athena-results/"
    athena_query(athena, f"CREATE DATABASE IF NOT EXISTS {DATABASE}", out)
    cols = ",\n  ".join(f"{c} string" for c in raw_column_names())
    athena_query(athena, f"""CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.raw_encounters (
  {cols})
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
LOCATION 's3://{bucket}/raw/diabetic_data/'
TBLPROPERTIES ('skip.header.line.count'='1')""", out)
    features = (ROOT / "sql/features.sql").read_text().replace("FROM raw_encounters", f"FROM {DATABASE}.raw_encounters")
    features = "\n".join(l for l in features.splitlines() if not l.strip().startswith("--"))
    rows = athena_query(athena, CHECK_SQL.format(features=features), out)
    return dict(zip(rows[0], rows[1]))


def local_check() -> dict:
    import duckdb
    from src.prepare import RAW, raw_column_names
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw_encounters AS SELECT * FROM read_csv('{RAW}', header=true, "
                f"all_varchar=true, names={raw_column_names()})")
    rel = con.execute(CHECK_SQL.format(features=(ROOT / "sql/features.sql").read_text()))
    names = [d[0] for d in rel.description]
    return {k: str(v) for k, v in zip(names, rel.fetchone())}


# ---------- 4/5. IAM + Lambda + logs ----------
def ensure_role(iam) -> str:
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "sts:AssumeRole",
                                                      "Principal": {"Service": "lambda.amazonaws.com"}}]}
    try:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust),
                              Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])["Role"]["Arn"]
        iam.attach_role_policy(RoleName=ROLE,
                               PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
        time.sleep(10)  # new roles take a few seconds before Lambda can use them
    return arn


def ensure_log_group(logs) -> None:
    try:
        logs.create_log_group(logGroupName=LOG_GROUP, tags=TAGS)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=7)


def ensure_function(lam, role_arn: str, bucket: str, key: str, version: str) -> str:
    cfg = dict(Runtime="python3.12", Handler="handler.lambda_handler", Role=role_arn,
               Architectures=["arm64"], MemorySize=512, Timeout=15,
               Environment={"Variables": {"MODEL_VERSION": version}})
    try:
        lam.get_function(FunctionName=FUNCTION)
        lam.update_function_code(FunctionName=FUNCTION, S3Bucket=bucket, S3Key=key, Architectures=["arm64"])
        lam.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION)
        lam.update_function_configuration(FunctionName=FUNCTION, **{k: v for k, v in cfg.items() if k != "Architectures"})
    except lam.exceptions.ResourceNotFoundException:
        lam.create_function(FunctionName=FUNCTION, Code={"S3Bucket": bucket, "S3Key": key}, Tags=TAGS, **cfg)
    lam.get_waiter("function_active_v2").wait(FunctionName=FUNCTION)
    lam.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION)
    return lam.get_function(FunctionName=FUNCTION)["Configuration"]["FunctionArn"]


# ---------- 6. smoke test ----------
def smoke_test(lam, n: int = 20) -> dict:
    import csv
    import lightgbm as lgb
    import pandas as pd
    from src.encoding import encode_frame, load_schema
    from src.features import build_features
    from src.prepare import RAW

    feats = pd.read_parquet(ROOT / "data/processed/features.parquet")
    ids = set(feats.loc[feats.split == "test", "encounter_id"].head(n))
    with open(RAW, newline="") as f:
        records = [r for r in csv.DictReader(f) if int(r["encounter_id"]) in ids]
    booster = lgb.Booster(model_file=str(ROOT / "artifacts/model.txt"))
    expected = booster.predict(encode_frame(pd.DataFrame([build_features(r) for r in records]),
                                            load_schema(ROOT / "artifacts/schema.json")))
    gaps, latencies = [], []
    for rec, exp in zip(records, expected):
        t = time.perf_counter()
        resp = lam.invoke(FunctionName=FUNCTION, Payload=json.dumps(rec).encode())
        latencies.append((time.perf_counter() - t) * 1000)
        body = json.loads(json.loads(resp["Payload"].read())["body"])
        gaps.append(abs(body["risk"] - exp))
    return {"invocations": len(records), "max_risk_difference_vs_local": round(max(gaps), 6),
            "first_call_ms": round(latencies[0]), "median_warm_ms": round(sorted(latencies[1:])[len(latencies[1:]) // 2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="where the $1 budget alert goes")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    bucket = f"{PROJECT}-{account}-{args.region}"
    state = {**load_state(), "account": account, "region": args.region, "bucket": bucket,
             "function": FUNCTION, "role": ROLE, "log_group": LOG_GROUP, "database": DATABASE,
             "budget": f"{PROJECT}-1-dollar"}
    save_state(state)

    print("1/6 budget alarm ($1)");          ensure_budget(session, account, args.email)
    s3 = session.client("s3")
    print("2/6 S3 bucket + uploads");        ensure_bucket(s3, bucket, args.region)
    s3.upload_file(str(ROOT / "data/raw/diabetic_data.csv"), bucket, "raw/diabetic_data/diabetic_data.csv")
    version = time.strftime("%Y%m%d-%H%M%S")
    key = f"lambda/{FUNCTION}-{version}.zip"
    s3.put_object(Bucket=bucket, Key=key, Body=zip_lambda())

    print("3/6 Athena: same SQL as local");  athena_totals = setup_athena(session.client("athena"), bucket)
    local_totals = local_check()
    state["athena_vs_local"] = {"athena": athena_totals, "local": local_totals,
                                "match": athena_totals == local_totals}
    print("    athena:", athena_totals); print("    local: ", local_totals)

    print("4/6 IAM role + Lambda");          role_arn = ensure_role(session.client("iam"))
    print("5/6 log group (7-day retention)"); ensure_log_group(session.client("logs"))
    state["function_arn"] = ensure_function(session.client("lambda"), role_arn, bucket, key, version)
    save_state(state)

    print("6/6 smoke test");                 state["smoke_test"] = smoke_test(session.client("lambda"))
    print("   ", state["smoke_test"])
    save_state(state)
    print(f"\nDone. State in {STATE.relative_to(ROOT)}. Remove everything with: make destroy")


if __name__ == "__main__":
    main()
