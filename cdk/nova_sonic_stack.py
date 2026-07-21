"""
CDK stack for the Nova Sonic variant of the Spanish tutor.

WHY THIS IS DIFFERENT FROM PronunciationCheckerStack
----------------------------------------------------
The `dev` branch pipeline is: Transcribe (STT) -> Wav2Vec2 on SageMaker (phonemes)
-> Claude streaming (conversation) -> Polly (TTS), fanned out over an API Gateway
WebSocket + Lambda. Four services, plus a 24/7 GPU endpoint.

Nova Sonic collapses STT + reasoning + TTS into ONE speech-to-speech model with
native turn-taking and barge-in. There is no Transcribe, no Polly, and (by default)
no SageMaker GPU endpoint.

The one hard architectural constraint: Nova Sonic uses the
`InvokeModelWithBidirectionalStream` API, a persistent HTTP/2 stream. Lambda cannot
hold that stream, so the dev branch's serverless WS+Lambda model does NOT work here.
Instead we run a small always-on **Fargate bridge**: browsers connect to it over a
WebSocket (through an ALB), and it relays audio to/from the Nova Sonic bidirectional
stream. It also resolves tool calls (course-material search via the Bedrock Knowledge
Base, and non-interrupting pronunciation/grammar feedback).

Reused from the HTTP stack: the materials S3 bucket, the Canvas sync Lambda, and the
Bedrock Knowledge Base wiring.
"""

import os

from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr_assets as ecr_assets,
    aws_lambda as _lambda,
    aws_events as events,
    aws_events_targets as targets,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

HERE = os.path.dirname(__file__)

# Nova 2 Sonic. Override via context/env if you need Nova Sonic v1 ("amazon.nova-sonic-v1:0")
# or a different region. Nova Sonic is region-limited; us-east-1 is the safest default.
NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"
NOVA_SONIC_REGION = "us-east-1"


class NovaSonicTutorStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # --- Materials bucket (S3 data source for the Knowledge Base) ---
        materials_bucket = s3.Bucket(self, "MaterialsBucket",
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        # --- Canvas API token + sync Lambda (unchanged from the HTTP stack) ---
        canvas_secret = secretsmanager.Secret(self, "CanvasApiToken",
            secret_name="tutor-sonic/canvas-api-token",
            description="Canvas LMS API token for syncing course materials")

        sync_lambda = _lambda.Function(self, "CanvasSyncHandler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="sync.lambda_handler",
            code=_lambda.Code.from_asset(os.path.join(HERE, "..", "lambda")),
            timeout=Duration.seconds(300),
            memory_size=512,
            environment={
                "MATERIALS_BUCKET": materials_bucket.bucket_name,
                "CANVAS_SECRET_ARN": canvas_secret.secret_arn,
                "KNOWLEDGE_BASE_ID": "SET_VIA_SETUP_KB",
            })
        materials_bucket.grant_write(sync_lambda)
        canvas_secret.grant_read(sync_lambda)
        sync_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:StartIngestionJob", "bedrock:ListDataSources", "bedrock:ListKnowledgeBases"],
            resources=["*"]))

        events.Rule(self, "SyncSchedule",
            schedule=events.Schedule.rate(Duration.hours(6)),
            targets=[targets.LambdaFunction(sync_lambda)])

        # --- System prompt for the speech-to-speech tutor ---
        with open(os.path.join(HERE, "..", "prompts", "system_sonic.txt")) as f:
            system_prompt = f.read()

        # --- Nova Sonic WebSocket bridge on Fargate ---
        vpc = ec2.Vpc(self, "TutorVpc", max_azs=2, nat_gateways=1)
        cluster = ecs.Cluster(self, "TutorCluster", vpc=vpc)

        bridge_image = ecr_assets.DockerImageAsset(self, "NovaSonicBridgeImage",
            directory=os.path.join(HERE, "..", "nova_sonic_container"),
            platform=ecr_assets.Platform.LINUX_AMD64)

        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "NovaSonicBridge",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            public_load_balancer=True,
            # WebSocket connections are long-lived; raise the ALB idle timeout well above
            # the 60s default so idle pauses in conversation don't drop the socket.
            idle_timeout=Duration.seconds(3600),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(bridge_image),
                container_port=8080,
                environment={
                    "NOVA_SONIC_MODEL_ID": NOVA_SONIC_MODEL_ID,
                    "BEDROCK_REGION": NOVA_SONIC_REGION,
                    "KNOWLEDGE_BASE_ID": "SET_VIA_SETUP_KB",
                    "MATERIALS_BUCKET": materials_bucket.bucket_name,
                    "SYSTEM_PROMPT": system_prompt,
                },
            ),
        )

        # ALB health check hits the bridge's /health endpoint.
        service.target_group.configure_health_check(
            path="/health", healthy_http_codes="200")

        task_role = service.task_definition.task_role
        # Nova Sonic bidirectional streaming.
        task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModelWithBidirectionalStream", "bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{NOVA_SONIC_REGION}::foundation-model/amazon.nova-sonic-v1:0",
                f"arn:aws:bedrock:{NOVA_SONIC_REGION}::foundation-model/amazon.nova-2-sonic-v1:0",
            ]))
        # Knowledge Base retrieval for the search_materials tool.
        task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["bedrock:Retrieve"],
            resources=[f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/*"]))
        materials_bucket.grant_read(task_role)

        # Optional autoscaling: one browser session pins one connection to one task, so
        # scale on CPU as concurrent conversations grow.
        scaling = service.service.auto_scale_task_count(min_capacity=1, max_capacity=5)
        scaling.scale_on_cpu_utilization("CpuScaling", target_utilization_percent=65)

        # --- Outputs ---
        CfnOutput(self, "WsUrl",
            value=f"ws://{service.load_balancer.load_balancer_dns_name}/ws",
            description="WebSocket endpoint for the browser client. NOTE: this is ws:// "
                        "(no TLS). Browsers block mic access + ws:// from https pages; "
                        "put HTTPS/wss in front (ACM cert on the ALB) for production.")
        CfnOutput(self, "AlbDns", value=service.load_balancer.load_balancer_dns_name)
        CfnOutput(self, "MaterialsBucketName", value=materials_bucket.bucket_name)
