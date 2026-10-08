"""
Remove everything aws/deploy.py created. Safe to run twice; missing pieces are skipped.

  python -m aws.destroy                 # keeps the $1 budget alarm (it costs nothing)
  python -m aws.destroy --delete-budget # removes that too

Ends by listing anything still tagged project=readmission-risk, so you can see it's clean.
"""
import argparse
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "aws/state.json"


def quiet(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NoSuchEntity", "ResourceNotFoundException", "NoSuchBucket", "EntityNotFoundException",
                    "NotFoundException", "404"):
            return False
        raise


def empty_bucket(s3, bucket: str) -> None:
    try:
        pages = s3.get_paginator("list_object_versions").paginate(Bucket=bucket)
        for page in pages:
            objs = [{"Key": o["Key"], "VersionId": o["VersionId"]}
                    for o in page.get("Versions", []) + page.get("DeleteMarkers", [])]
            if objs:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchBucket":
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-budget", action="store_true")
    args = ap.parse_args()
    if not STATE.exists():
        raise SystemExit("aws/state.json not found: nothing was deployed from this folder")
    st = json.loads(STATE.read_text())
    session = boto3.Session(region_name=st["region"])

    lam, logs, iam, glue, s3 = (session.client(n) for n in ("lambda", "logs", "iam", "glue", "s3"))
    print("Lambda function:", quiet(lam.delete_function, FunctionName=st["function"]))
    print("Log group:      ", quiet(logs.delete_log_group, logGroupName=st["log_group"]))
    quiet(iam.detach_role_policy, RoleName=st["role"],
          PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    print("IAM role:       ", quiet(iam.delete_role, RoleName=st["role"]))
    quiet(glue.delete_table, DatabaseName=st["database"], Name="raw_encounters")
    print("Athena database:", quiet(glue.delete_database, Name=st["database"]))
    empty_bucket(s3, st["bucket"])
    print("S3 bucket:      ", quiet(s3.delete_bucket, Bucket=st["bucket"]))
    if args.delete_budget:
        b = session.client("budgets", region_name="us-east-1")
        print("Budget:         ", quiet(b.delete_budget, AccountId=st["account"], BudgetName=st["budget"]))

    left = session.client("resourcegroupstaggingapi").get_resources(
        TagFilters=[{"Key": "project", "Values": ["readmission-risk"]}])["ResourceTagMappingList"]
    print("\nStill tagged project=readmission-risk:", [r["ResourceARN"] for r in left] or "nothing")
    STATE.rename(STATE.with_suffix(".destroyed.json"))


if __name__ == "__main__":
    main()
