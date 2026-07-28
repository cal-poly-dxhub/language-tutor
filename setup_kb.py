#!/usr/bin/env python3
"""
Post-deploy setup: wire a Bedrock Knowledge Base ID into the deployed app.

CloudFormation's Bedrock KB support needs a pre-existing vector store, while the
console's "Quick create" builds one for you — so the KB is created by hand once and
its ID is injected here.

    1. ./deploy.sh                         (see README)
    2. create the KB in the console        (run this script with no args for the steps)
    3. python3 setup_kb.py <KB_ID>

This updates the Canvas sync Lambda's environment AND the Fargate bridge's task
definition, then redeploys the service so the running container picks up the new value.
"""

import argparse
import sys

import boto3

DEFAULT_STACK = "LanguageTutorStack"
DEFAULT_REGION = "us-west-2"


def stack_outputs(region, stack):
    cf = boto3.client("cloudformation", region_name=region)
    resp = cf.describe_stacks(StackName=stack)
    return {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}


def print_instructions(region, stack):
    outputs = {}
    try:
        outputs = stack_outputs(region, stack)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read stack {stack} in {region}: {exc}\n")
    bucket = outputs.get("MaterialsBucketName", "<MaterialsBucketName output>")

    print("Adding a Knowledge Base to the tutor\n")
    print("1. Put some course material in the bucket the Knowledge Base will index.")
    print("   An empty bucket produces an empty Knowledge Base, and the tutor will")
    print("   correctly tell learners it has nothing to look up:\n")
    print(f"     aws s3 cp data/syllabus.pdf s3://{bucket}/ --region {region}")
    print(f"     aws s3 ls s3://{bucket}/ --region {region}\n")
    print(f"2. Create the Knowledge Base IN {region} — it must share the bridge's")
    print("   region, or retrieval fails at conversation time:\n")
    print(f"     https://{region}.console.aws.amazon.com/bedrock/home"
          f"?region={region}#/knowledge-bases/create\n")
    print("   - Name: tutor-materials")
    print("   - Vector store: 'Quick create a new vector store' (OpenSearch Serverless)")
    print(f"   - Data source: S3 -> s3://{bucket}")
    print("   - Embeddings model: Titan Text Embeddings V2\n")
    print("3. Select the data source and choose Sync. Nothing is retrievable until the")
    print("   first sync finishes.\n")
    print("4. Copy the Knowledge Base ID and wire it into the app:\n")
    print(f"     python3 setup_kb.py <KB_ID> --region {region}\n")
    print("   That patches the sync Lambda and the bridge task definition, then rolls")
    print("   the service. Open browser sessions need reconnecting afterwards.\n")
    print("5. Check it took effect. Ask the tutor about the syllabus, or send")
    print('   {"type":"ping"} on the WebSocket and confirm "knowledgeBase" is not null.')
    if outputs.get("SyncFunctionName"):
        print("\nOptional — Canvas LMS: put your token in the secret, then force a sync:")
        print("     aws secretsmanager put-secret-value --secret-id "
              "language-tutor/canvas-api-token \\")
        print('       --secret-string \'{"base_url":"https://school.instructure.com",'
              '"token":"...","course_id":"12345"}\'')
        print(f"     aws lambda invoke --function-name "
              f"{outputs['SyncFunctionName']} /dev/null --region {region}")


def update_lambdas(region, prefixes, kb_id):
    client = boto3.client("lambda", region_name=region)
    functions = []
    for page in client.get_paginator("list_functions").paginate():
        functions.extend(page["Functions"])

    updated = 0
    for prefix in prefixes:
        for fn in functions:
            if not fn["FunctionName"].startswith(prefix):
                continue
            config = client.get_function_configuration(FunctionName=fn["FunctionName"])
            env = config.get("Environment", {}).get("Variables", {})
            if env.get("KNOWLEDGE_BASE_ID") == kb_id:
                print(f"  = {fn['FunctionName']} already set")
                updated += 1
                continue
            env["KNOWLEDGE_BASE_ID"] = kb_id
            client.update_function_configuration(
                FunctionName=fn["FunctionName"], Environment={"Variables": env})
            print(f"  + {fn['FunctionName']} -> KNOWLEDGE_BASE_ID={kb_id}")
            updated += 1
    return updated


def update_bridge(region, outputs, kb_id):
    """
    Patch KNOWLEDGE_BASE_ID into the bridge task definition and roll the service.

    ECS task definitions are immutable, so this registers a new revision with the env
    var replaced and points the service at it. The rolling deployment drops existing
    WebSocket sessions — reconnect the client afterwards.
    """
    service_name = outputs.get("BridgeServiceName")
    if not service_name:
        print("  (no BridgeServiceName output — skipping bridge update)")
        return

    ecs = boto3.client("ecs", region_name=region)
    clusters = ecs.list_clusters()["clusterArns"]
    target = None
    for cluster in clusters:
        found = ecs.list_services(cluster=cluster, maxResults=100)["serviceArns"]
        if any(arn.endswith(f"/{service_name}") for arn in found):
            target = cluster
            break
    if not target:
        print(f"  (service {service_name} not found in any cluster — skipping)")
        return

    svc = ecs.describe_services(cluster=target, services=[service_name])["services"][0]
    task_def = ecs.describe_task_definition(
        taskDefinition=svc["taskDefinition"])["taskDefinition"]

    changed = False
    for container in task_def["containerDefinitions"]:
        env = {e["name"]: e["value"] for e in container.get("environment", [])}
        if env.get("KNOWLEDGE_BASE_ID") == kb_id:
            continue
        env["KNOWLEDGE_BASE_ID"] = kb_id
        container["environment"] = [{"name": k, "value": v} for k, v in env.items()]
        changed = True

    if not changed:
        print(f"  = {service_name} already set")
        return

    # Re-register with only the fields RegisterTaskDefinition accepts. describe returns
    # several read-only fields (taskDefinitionArn, revision, status, requiresAttributes,
    # compatibilities, registeredAt/By, deregisteredAt) that the register call rejects,
    # and a whitelist stays correct if the API adds more.
    registerable = {
        "family", "taskRoleArn", "executionRoleArn", "networkMode",
        "containerDefinitions", "volumes", "placementConstraints",
        "requiresCompatibilities", "cpu", "memory", "tags", "pidMode", "ipcMode",
        "proxyConfiguration", "inferenceAccelerators", "ephemeralStorage",
        "runtimePlatform", "enableFaultInjection",
    }
    payload = {k: v for k, v in task_def.items() if k in registerable}

    new_arn = ecs.register_task_definition(**payload)["taskDefinition"][
        "taskDefinitionArn"]
    # ECS parameters are camelCase in boto3 (taskDefinition, not task_definition).
    ecs.update_service(cluster=target, service=service_name,
                       taskDefinition=new_arn, forceNewDeployment=True)
    print(f"  + {service_name} -> new task definition {new_arn.rsplit('/', 1)[-1]}")
    print("    rolling deploy started; existing WebSocket sessions will reconnect")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_id", nargs="?", help="Bedrock Knowledge Base ID")
    parser.add_argument("--stack", default=DEFAULT_STACK)
    parser.add_argument("--region", default=DEFAULT_REGION)
    args = parser.parse_args()

    if not args.kb_id:
        print_instructions(args.region, args.stack)
        return 0

    outputs = stack_outputs(args.region, args.stack)
    print(f"Wiring KNOWLEDGE_BASE_ID={args.kb_id} into {args.stack} ({args.region})")

    if not update_lambdas(args.region, [f"{args.stack}-CanvasSyncHandler"], args.kb_id):
        print("  (no matching Lambda functions found)")

    update_bridge(args.region, outputs, args.kb_id)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
