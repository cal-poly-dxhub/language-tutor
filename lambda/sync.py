"""Canvas LMS sync Lambda. Polls Canvas API, downloads materials to S3, triggers KB ingestion."""

import json
import os
import urllib.request
import urllib.parse

import boto3

MATERIALS_BUCKET = os.environ["MATERIALS_BUCKET"]
CANVAS_SECRET_ARN = os.environ["CANVAS_SECRET_ARN"]
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]

s3 = boto3.client("s3")
secrets = boto3.client("secretsmanager")
bedrock_agent = boto3.client("bedrock-agent")


def get_canvas_config():
    secret = json.loads(secrets.get_secret_value(SecretId=CANVAS_SECRET_ARN)["SecretString"])
    return secret["base_url"], secret["token"], secret["course_id"]


def canvas_get(base_url, token, path):
    url = f"{base_url}/api/v1/{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def download_file(base_url, token, file_url):
    req = urllib.request.Request(file_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def lambda_handler(event, context):
    base_url, token, course_id = get_canvas_config()

    # Get course files
    files = canvas_get(base_url, token, f"courses/{course_id}/files?per_page=100")

    synced = 0
    for f in files:
        content = download_file(base_url, token, f["url"])
        key = f"canvas/{course_id}/{f['display_name']}"
        s3.put_object(Bucket=MATERIALS_BUCKET, Key=key, Body=content)
        synced += 1

    # Trigger KB ingestion
    data_sources = bedrock_agent.list_data_sources(knowledgeBaseId=KNOWLEDGE_BASE_ID)
    for ds in data_sources.get("dataSourceSummaries", []):
        bedrock_agent.start_ingestion_job(
            knowledgeBaseId=KNOWLEDGE_BASE_ID, dataSourceId=ds["dataSourceId"])

    return {"statusCode": 200, "body": json.dumps({"synced": synced})}
