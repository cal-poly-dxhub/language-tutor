"""
Unified CDK stack for the two-mode Language Tutor web app.

ONE STACK, TWO INDEPENDENT MODES
--------------------------------
Mode A — Raw Nova Sonic. A single speech-to-speech model does STT + reasoning + TTS +
turn-taking + barge-in inside one bidirectional HTTP/2 stream. Lambda cannot hold that
stream, so it runs on an always-on **Fargate bridge** behind an ALB. No coaching, no
tools — just conversation.

Mode B — Custom bot. The project's differentiator: record one utterance, then
Transcribe (STT) -> Wav2Vec2 on SageMaker (phonemes) -> Claude (conversation + grammar/
pronunciation evaluation + course-material RAG) -> Polly (TTS). Runs on a **Lambda**
behind an HTTP API. Transcription happens server-side so the browser only uploads a WAV.

HOSTING
-------
One CloudFront distribution serves the static web client AND proxies:
  /ws                    -> the Fargate Sonic bridge (WebSocket)
  /converse, /upload-url -> the custom-bot HTTP API
So the page, the socket and the API share one https origin: no mixed-content problem
(https page + wss socket), no CORS, and no endpoint to paste. config.json (written here)
carries only the wss:// URL.

COST NOTE: the Wav2Vec2 phoneme endpoint is a 24/7 ml.g4dn.xlarge GPU instance and the
Fargate bridge is an always-on task. Deploy with `-c phonemes=false` to drop the GPU
endpoint; the custom bot then gives grammar feedback only.

SECURITY NOTE: the Sonic WebSocket is gated by a generated shared secret (see the
BridgeAccessToken output). The custom-bot HTTP API (/converse, /upload-url) is currently
UNAUTHENTICATED — anyone with the URL can spend Bedrock/Transcribe/Polly on your bill.
Add a Cognito/JWT authorizer (or a shared-secret Lambda authorizer) before exposing this
publicly.
"""

import os

from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput, BundlingOptions,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr_assets as ecr_assets,
    aws_sagemaker as sagemaker,
    aws_lambda as _lambda,
    aws_apigatewayv2 as apigwv2,
    aws_events as events,
    aws_events_targets as targets,
    aws_secretsmanager as secretsmanager,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
)
from constructs import Construct

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

# Nova Sonic is region-limited. us-west-2 matches the rest of this project's footprint;
# the Knowledge Base, Bedrock and Polly are all used from the same region.
NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"
NOVA_SONIC_VOICE = "lupe"
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class LanguageTutorStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # `-c phonemes=false` skips the GPU endpoint (see cost note).
        want_phonemes = str(self.node.try_get_context("phonemes")).lower() != "false"

        # ---- Course materials (S3 data source for the Knowledge Base) ----
        # The custom-bot flow has the browser talk to S3 directly via presigned URLs:
        # a PUT (record upload, with a non-simple Content-Type header -> CORS preflight)
        # and a GET (reply audio). Those calls hit s3.amazonaws.com, a different origin
        # than the CloudFront page, so the bucket needs CORS. Presigned signatures still
        # gate access; CORS only tells the browser the cross-origin call is allowed.
        materials_bucket = s3.Bucket(self, "MaterialsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT,
                                 s3.HttpMethods.HEAD],
                allowed_origins=["*"],
                allowed_headers=["*"],
                exposed_headers=["ETag"],
                max_age=3000)],
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        # ---- Canvas LMS sync (scheduled; no public write endpoint) ----
        canvas_secret = secretsmanager.Secret(self, "CanvasApiToken",
            secret_name="language-tutor/canvas-api-token",
            description="Canvas LMS API token for syncing course materials")

        sync_lambda = _lambda.Function(self, "CanvasSyncHandler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="sync.lambda_handler",
            code=_lambda.Code.from_asset(os.path.join(ROOT, "lambda")),
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
            actions=["bedrock:StartIngestionJob", "bedrock:ListDataSources",
                     "bedrock:ListKnowledgeBases"],
            resources=["*"]))
        events.Rule(self, "SyncSchedule",
            schedule=events.Schedule.rate(Duration.hours(6)),
            targets=[targets.LambdaFunction(sync_lambda)])

        # ---- Mode B: Wav2Vec2 phoneme endpoint (optional) ----
        endpoint_name = ""
        phoneme_endpoint_arn = None
        if want_phonemes:
            sagemaker_role = iam.Role(self, "SageMakerRole",
                assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
                managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess")])

            model_image = ecr_assets.DockerImageAsset(self, "Wav2Vec2Image",
                directory=os.path.join(ROOT, "container"),
                platform=ecr_assets.Platform.LINUX_AMD64)
            model_image.repository.grant_pull(sagemaker_role)

            model = sagemaker.CfnModel(self, "PhonemeModel",
                execution_role_arn=sagemaker_role.role_arn,
                primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                    image=model_image.image_uri, mode="SingleModel"))

            endpoint_config = sagemaker.CfnEndpointConfig(self, "EndpointConfig",
                production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic", model_name=model.attr_model_name,
                    initial_instance_count=1, instance_type="ml.g4dn.xlarge",
                    initial_variant_weight=1.0)])
            endpoint_config.add_dependency(model)

            endpoint = sagemaker.CfnEndpoint(self, "PhonemeEndpoint",
                endpoint_config_name=endpoint_config.attr_endpoint_config_name)
            endpoint.add_dependency(endpoint_config)

            endpoint_name = endpoint.attr_endpoint_name
            phoneme_endpoint_arn = (
                f"arn:aws:sagemaker:{self.region}:{self.account}:"
                f"endpoint/{endpoint.attr_endpoint_name}")

        # ---- Mode B: custom-bot conversation Lambda + HTTP API ----
        with open(os.path.join(ROOT, "prompts", "system.txt")) as f:
            custom_prompt = f.read()

        # The converse Lambda needs the amazon-transcribe SDK (server-side STT), which
        # pulls in awscrt's native extension (_awscrt.abi3.so). That wheel is
        # architecture-specific, so we must fetch the wheel that matches the Lambda
        # runtime (x86_64) rather than whatever the build host happens to be. Building
        # on an Apple Silicon (arm64) Mac would otherwise install arm64 wheels that fail
        # to load on the x86_64 Lambda ("cannot open shared object file"). Pinning
        # --platform/--python-version/--only-binary makes pip download the manylinux
        # x86_64 wheels without executing them, so the build is host-arch independent.
        converse_code = _lambda.Code.from_asset(
            os.path.join(ROOT, "lambda"),
            bundling=BundlingOptions(
                image=_lambda.Runtime.PYTHON_3_12.bundling_image,
                command=["bash", "-c",
                         "pip install -r requirements.txt -t /asset-output "
                         "--platform manylinux2014_x86_64 --implementation cp "
                         "--python-version 3.12 --only-binary=:all: && "
                         "cp -au . /asset-output"],
            ))

        converse_role = iam.Role(self, "ConverseRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole")])
        converse_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[
                "arn:aws:bedrock:*::foundation-model/anthropic.*",
                f"arn:aws:bedrock:*:{self.account}:inference-profile/us.anthropic.*",
            ]))
        converse_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:Retrieve"],
            resources=[f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/*"]))
        converse_role.add_to_policy(iam.PolicyStatement(
            actions=["polly:SynthesizeSpeech"], resources=["*"]))
        # Transcribe streaming has no resource-level scoping.
        converse_role.add_to_policy(iam.PolicyStatement(
            actions=["transcribe:StartStreamTranscription"], resources=["*"]))
        converse_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject"],
            resources=[f"{materials_bucket.bucket_arn}/*"]))
        if phoneme_endpoint_arn:
            converse_role.add_to_policy(iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint"],
                resources=[phoneme_endpoint_arn]))

        converse_env = {
            "BEDROCK_MODEL_ID": BEDROCK_MODEL_ID,
            "SYSTEM_PROMPT": custom_prompt,
            "KNOWLEDGE_BASE_ID": "SET_VIA_SETUP_KB",
            "MATERIALS_BUCKET": materials_bucket.bucket_name,
            "TRANSCRIBE_LANGUAGE": "es-US",
            "POLLY_VOICE": "Lupe",
            "POLLY_LANGUAGE": "es-US",
        }
        if endpoint_name:
            converse_env["SAGEMAKER_ENDPOINT"] = endpoint_name

        converse_handler = _lambda.Function(self, "ConverseHandler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=converse_code,
            role=converse_role,
            architecture=_lambda.Architecture.X86_64,  # must match the awscrt wheels bundled above
            timeout=Duration.seconds(60),
            memory_size=1024,
            environment=converse_env)

        http_api = apigwv2.CfnApi(self, "CustomBotApi",
            name="LanguageTutorApi", protocol_type="HTTP",
            cors_configuration=apigwv2.CfnApi.CorsProperty(
                allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Content-Type"]))

        converse_integration = apigwv2.CfnIntegration(self, "ConverseIntegration",
            api_id=http_api.ref, integration_type="AWS_PROXY", integration_method="POST",
            integration_uri=converse_handler.function_arn, payload_format_version="2.0")

        for rid, route_key in [
            ("UploadRoute", "GET /upload-url"),
            ("ConverseRoute", "POST /converse"),
        ]:
            apigwv2.CfnRoute(self, rid, api_id=http_api.ref, route_key=route_key,
                target=f"integrations/{converse_integration.ref}")

        apigwv2.CfnStage(self, "ApiStage",
            api_id=http_api.ref, stage_name="$default", auto_deploy=True)

        converse_handler.add_permission("HttpApiInvoke",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=f"arn:aws:execute-api:{self.region}:{self.account}:{http_api.ref}/*")

        api_domain = f"{http_api.ref}.execute-api.{self.region}.amazonaws.com"

        # ---- Mode A: Nova Sonic bridge on Fargate ----
        with open(os.path.join(ROOT, "prompts", "system_sonic.txt")) as f:
            sonic_prompt = f.read()

        access_token = secretsmanager.Secret(self, "BridgeAccessToken",
            secret_name="language-tutor/bridge-access-token",
            description="Shared secret required to open a raw-Sonic WebSocket session",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=40, exclude_punctuation=True))

        vpc = ec2.Vpc(self, "TutorVpc", max_azs=2, nat_gateways=1)
        cluster = ecs.Cluster(self, "TutorCluster", vpc=vpc)

        bridge_image = ecr_assets.DockerImageAsset(self, "SonicBridgeImage",
            directory=os.path.join(ROOT, "sonic_container"),
            platform=ecr_assets.Platform.LINUX_AMD64)

        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "SonicBridge",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            public_load_balancer=True,
            idle_timeout=Duration.seconds(3600),   # long-lived WebSockets
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(bridge_image),
                container_port=8080,
                environment={
                    "NOVA_SONIC_MODEL_ID": NOVA_SONIC_MODEL_ID,
                    "NOVA_SONIC_VOICE": NOVA_SONIC_VOICE,
                    "BEDROCK_REGION": self.region,
                    "SYSTEM_PROMPT": sonic_prompt,
                },
                secrets={"ACCESS_TOKEN": ecs.Secret.from_secrets_manager(access_token)},
            ),
        )
        service.target_group.configure_health_check(
            path="/health", healthy_http_codes="200",
            interval=Duration.seconds(30), timeout=Duration.seconds(5))
        service.target_group.set_attribute("deregistration_delay.timeout_seconds", "30")

        task_role = service.task_definition.task_role
        # Both actions on purpose: bedrock evaluates InvokeModel alongside the streaming
        # action for this API, and granting only the streaming one fails silently at
        # stream teardown.
        task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModelWithBidirectionalStream", "bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{self.region}::foundation-model/{NOVA_SONIC_MODEL_ID}",
                f"arn:aws:bedrock:{self.region}::foundation-model/amazon.nova-sonic-v1:0",
            ]))

        scaling = service.service.auto_scale_task_count(min_capacity=1, max_capacity=5)
        scaling.scale_on_cpu_utilization("CpuScaling", target_utilization_percent=65)

        # ---- Hosting: CloudFront over the web client + both backends ----
        site_bucket = s3.Bucket(self, "SiteBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        api_origin = origins.HttpOrigin(api_domain,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
            read_timeout=Duration.seconds(60))

        api_behavior = cloudfront.BehaviorOptions(
            origin=api_origin,
            allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
            cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
            origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
            viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
        )

        distribution = cloudfront.Distribution(self, "SiteDistribution",
            comment="Language Tutor web client + Sonic bridge + custom-bot API",
            default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            additional_behaviors={
                # WebSocket upgrade to the Fargate bridge. Nothing cached; upgrade
                # headers + ?token= must reach the origin.
                "/ws": cloudfront.BehaviorOptions(
                    origin=origins.LoadBalancerV2Origin(
                        service.load_balancer,
                        protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                        read_timeout=Duration.seconds(60),
                        keepalive_timeout=Duration.seconds(60),
                    ),
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                ),
                # Custom-bot HTTP API, same origin as the page (no CORS needed).
                "/converse": api_behavior,
                "/upload-url": api_behavior,
            },
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
        )

        # Publish frontend/ on every deploy; generate config.json here so the client
        # never has to be told where the bridge is.
        s3deploy.BucketDeployment(self, "SiteContent",
            sources=[
                s3deploy.Source.asset(os.path.join(ROOT, "frontend")),
                s3deploy.Source.json_data("config.json", {
                    "wsUrl": f"wss://{distribution.distribution_domain_name}/ws",
                    "defaultMode": "sonic",
                }),
            ],
            destination_bucket=site_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
            cache_control=[
                s3deploy.CacheControl.set_public(),
                s3deploy.CacheControl.max_age(Duration.seconds(0)),
                s3deploy.CacheControl.must_revalidate(),
            ])

        # ---- Outputs ----
        CfnOutput(self, "AppUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="Open this. Mic works because CloudFront serves over https.")
        CfnOutput(self, "WsUrl",
            value=f"wss://{distribution.distribution_domain_name}/ws",
            description="Raw-Sonic WebSocket through CloudFront. Append ?token=<secret>. "
                        "The web client fills this in itself.")
        CfnOutput(self, "ApiUrl",
            value=f"https://{api_domain}",
            description="Custom-bot HTTP API (also reachable same-origin via CloudFront).")
        CfnOutput(self, "AccessTokenSecretName", value=access_token.secret_name,
            description="aws secretsmanager get-secret-value --secret-id <this> "
                        "--query SecretString --output text")
        CfnOutput(self, "MaterialsBucketName", value=materials_bucket.bucket_name)
        CfnOutput(self, "SyncFunctionName", value=sync_lambda.function_name,
            description="Force a Canvas sync: aws lambda invoke --function-name <this> /dev/null")
        CfnOutput(self, "PhonemeEndpointName",
            value=endpoint_name or "disabled (-c phonemes=false)")
        CfnOutput(self, "BridgeServiceName", value=service.service.service_name)
