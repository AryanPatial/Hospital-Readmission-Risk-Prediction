"""
Run the AWS create/destroy code against moto (a local fake AWS), so the scripts are
checked without an account or any cost. Athena and the real Lambda runtime can't be
faked meaningfully; those are checked by the live smoke test in aws/deploy.py.
"""
import io
import json
import sys
import zipfile

import boto3
import pytest
from moto import mock_aws

from aws import deploy, destroy

REGION = "us-east-1"


@pytest.fixture
def aws_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("MOTO_IAM_LOAD_MANAGED_POLICIES", "true")  # so AWSLambdaBasicExecutionRole exists
    monkeypatch.setattr(deploy, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(destroy, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    with mock_aws():
        yield boto3.Session(region_name=REGION)


def tiny_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("handler.py", "def lambda_handler(e, c):\n    return e\n")
    return buf.getvalue()


def test_deploy_then_destroy_leaves_nothing(aws_env):
    s = aws_env
    account = s.client("sts").get_caller_identity()["Account"]
    bucket = f"readmission-risk-{account}-{REGION}"

    deploy.ensure_budget(s, account, "me@example.com")
    deploy.ensure_budget(s, account, "me@example.com")          # re-run is safe
    s3 = s.client("s3")
    deploy.ensure_bucket(s3, bucket, REGION)
    deploy.ensure_bucket(s3, bucket, REGION)
    s3.put_object(Bucket=bucket, Key="lambda/code.zip", Body=tiny_zip())
    s3.put_object(Bucket=bucket, Key="raw/diabetic_data/diabetic_data.csv", Body=b"a,b\n1,2\n")
    role_arn = deploy.ensure_role(s.client("iam"))
    deploy.ensure_log_group(s.client("logs"))
    deploy.ensure_log_group(s.client("logs"))
    deploy.ensure_function(s.client("lambda"), role_arn, bucket, "lambda/code.zip", "test")
    s.client("glue").create_database(DatabaseInput={"Name": deploy.DATABASE})

    # what deploy.main would have saved
    deploy.save_state({"account": account, "region": REGION, "bucket": bucket, "function": deploy.FUNCTION,
                       "role": deploy.ROLE, "log_group": deploy.LOG_GROUP, "database": deploy.DATABASE,
                       "budget": "readmission-risk-1-dollar"})

    pab = s3.get_public_access_block(Bucket=bucket)["PublicAccessBlockConfiguration"]
    assert all(pab.values()), "bucket must block all public access"
    budgets = s.client("budgets").describe_budgets(AccountId=account)["Budgets"]
    assert [b["BudgetLimit"]["Amount"] for b in budgets] in (["1"], ["1.0"])
    assert s.client("logs").describe_log_groups()["logGroups"][0]["retentionInDays"] == 7
    cfg = s.client("lambda").get_function(FunctionName=deploy.FUNCTION)["Configuration"]
    assert cfg["Architectures"] == ["arm64"] and cfg["Runtime"] == "python3.12"

    sys.argv = ["destroy", "--delete-budget"]
    destroy.main()

    assert s3.list_buckets()["Buckets"] == []
    assert s.client("lambda").list_functions()["Functions"] == []
    assert s.client("logs").describe_log_groups()["logGroups"] == []
    assert not [r for r in s.client("iam").list_roles()["Roles"] if r["RoleName"] == deploy.ROLE]
    assert s.client("glue").get_databases()["DatabaseList"] == []
    assert s.client("budgets").describe_budgets(AccountId=account).get("Budgets", []) == []


def test_destroy_without_state_refuses(aws_env):
    sys.argv = ["destroy"]
    with pytest.raises(SystemExit):
        destroy.main()
