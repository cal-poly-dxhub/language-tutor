#!/usr/bin/env python3
"""
Post-deploy setup: Create Knowledge Base via Bedrock console (Quick Create).

CloudFormation's Bedrock KB support requires pre-existing vector stores.
The console's "Quick Create" handles everything automatically.

Steps:
  1. Deploy CDK stack: CDK_DOCKER=finch npx cdk deploy
  2. Create KB in console (see instructions below)
  3. Run: python3 setup_kb.py <KNOWLEDGE_BASE_ID>

This script updates the Lambda env vars to wire in the KB ID.
"""

import sys
import boto3

REGION = "us-west-2"
STACK_NAME = "PronunciationCheckerStack"


def main():
    if len(sys.argv) < 2:
        cf = boto3.client("cloudformation", region_name=REGION)
        resp = cf.describe_stacks(StackName=STACK_NAME)
        bucket = next(o["OutputValue"] for o in resp["Stacks"][0]["Outputs"] if o["OutputKey"] == "MaterialsBucketName")

        print("Create the Knowledge Base in the Bedrock console:")
        print(f"  1. Open: https://{REGION}.console.aws.amazon.com/bedrock/home#/knowledge-bases/create")
        print(f"  2. Name: tutor-materials")
        print(f"  3. Use 'Quick create new vector store' (creates managed OpenSearch)")
        print(f"  4. Add S3 data source → s3://{bucket}")
        print(f"  5. Embedding model: Titan Text Embeddings V2")
        print(f"  6. After creation, copy the Knowledge Base ID")
        print(f"\n  Then run: python3 setup_kb.py <KB_ID>")
        return

    kb_id = sys.argv[1]
    lambda_client = boto3.client("lambda", region_name=REGION)

    # Update both Lambdas with the KB ID
    for fn_prefix in ["PronunciationCheckerStack-PronunciationHandler", "PronunciationCheckerStack-CanvasSyncHandler"]:
        functions = lambda_client.list_functions()["Functions"]
        for fn in functions:
            if fn["FunctionName"].startswith(fn_prefix):
                config = lambda_client.get_function_configuration(FunctionName=fn["FunctionName"])
                env = config.get("Environment", {}).get("Variables", {})
                env["KNOWLEDGE_BASE_ID"] = kb_id
                lambda_client.update_function_configuration(
                    FunctionName=fn["FunctionName"], Environment={"Variables": env})
                print(f"Updated {fn['FunctionName']} with KNOWLEDGE_BASE_ID={kb_id}")

    print("Done! Knowledge Base is now wired in.")


if __name__ == "__main__":
    main()
