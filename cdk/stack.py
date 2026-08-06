"""
CDK stack for the hybrid Language Tutor.

TWO LANES, ONE STACK
--------------------
Fast lane — conversation. Nova 2 Sonic is a speech-to-speech model: it does STT,
reasoning, TTS, turn detection and barge-in inside a single bidirectional stream.
That stream (`InvokeModelWithBidirectionalStream`) is a persistent HTTP/2 connection,
and Lambda cannot hold one open, so the conversation runs through an always-on
**Fargate bridge** behind an ALB instead of the serverless WS+Lambda path on `dev`.

Slow lane — coaching. Nova Sonic reports no phoneme detail, so the project's
differentiator stays: the bridge tees the learner's audio to the **Wav2Vec2 SageMaker
endpoint** for phoneme scoring, then asks **Claude Haiku** whether anything is worth
saying. Notes are pushed to the client as on-screen annotations, out of band, so the
spoken conversation is never interrupted.

Carried over from the `dev` stack: the materials bucket, the Canvas sync Lambda on a
schedule, and the Bedrock Knowledge Base wiring for course-material RAG.

COST NOTE: the phoneme endpoint is a 24/7 ml.g4dn.xlarge GPU instance, which dominates
the bill. Deploy with `-c phonemes=false` to drop it; the coach then reviews grammar
only and stays silent about pronunciation.
"""

import base64
import gzip
import json
import os

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_certificatemanager as acm,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_sagemaker as sagemaker,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

from cdk.language import (
    ProfileError,
    available_languages,
    load_profile,
    render_prompt,
)

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

# Nova Sonic is region-limited: us-east-1, us-west-2, ap-northeast-1 and eu-north-1 at
# the time of writing. us-west-2 is the default because it matches the rest of this
# project's footprint. The Knowledge Base and the coach model are used from the same
# region so no cross-region calls are needed.
NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"
NOVA_SONIC_REGION = "us-west-2"
# Haiku does all the pronunciation/grammar judgement, including deciding which turns
# deserve a note at all. Nothing is pre-filtered by keyword lists in code.
COACH_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# Which language is being taught. `-c language=french` switches the voice and every
# prompt; see languages/*.json.
DEFAULT_LANGUAGE = "spanish"


class LanguageTutorStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # `-c phonemes=false` skips the GPU endpoint (see cost note above).
        want_phonemes = str(self.node.try_get_context("phonemes")).lower() != "false"
        # `-c certificateArn=arn:...` puts TLS on the ALB so the client can use wss://
        # from an https page. Without it the listener is plain HTTP (see WsUrl output).
        certificate_arn = self.node.try_get_context("certificateArn")
        # `-c language=<name>` selects languages/<name>.json.
        language = self.node.try_get_context("language") or DEFAULT_LANGUAGE

        profile = load_profile(language)
        conversation_prompt = render_prompt("conversation.txt", profile)
        coach_prompt = render_prompt("coach.txt", profile)

        # Every language is shipped, not just the default, so the client can switch at
        # runtime without a redeploy. Rendered prompts total ~40KB across the seven
        # languages, which would crowd the 64KiB task-definition limit, so they travel
        # gzipped and base64-encoded in one variable (~8KB).
        bundles = {}
        for name in available_languages():
            other = load_profile(name)
            bundles[name] = {
                "label": other["targetLanguage"],
                "voice": other["voiceId"],
                "espeakLanguage": other["espeakLanguage"],
                "system": render_prompt("conversation.txt", other),
                "coach": render_prompt("coach.txt", other),
            }
        packed = base64.b64encode(
            gzip.compress(json.dumps(bundles, separators=(",", ":")).encode())).decode()
        if len(packed) > 24000:
            raise ProfileError(
                f"packed language bundles are {len(packed)} bytes, too large for a task "
                f"definition environment variable — move them to S3 or trim the prompts")

        # --- Course materials (S3 data source for the Knowledge Base) ---
        materials_bucket = s3.Bucket(self, "MaterialsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        # --- Canvas LMS sync (unchanged behaviour from the dev stack) ---
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

        # Scheduled sync only. The dev stack also exposed an unauthenticated
        # POST /sync route; a public write endpoint isn't worth it here, so a manual
        # sync is `aws lambda invoke --function-name <SyncFunctionName> /dev/null`.
        events.Rule(self, "SyncSchedule",
            schedule=events.Schedule.rate(Duration.hours(6)),
            targets=[targets.LambdaFunction(sync_lambda)])

        # --- Slow lane: Wav2Vec2 phoneme endpoint (optional) ---
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

        # --- Shared-secret gate for the bridge ---
        # The ALB is internet-facing. Without this, anyone who discovers the DNS name
        # can hold conversations on your Bedrock bill. The bridge requires this token
        # as ?token= or an Authorization: Bearer header on the WebSocket upgrade.
        access_token = secretsmanager.Secret(self, "BridgeAccessToken",
            secret_name="language-tutor/bridge-access-token",
            description="Shared secret required to open a tutor WebSocket session",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=40, exclude_punctuation=True))

        # --- Fast lane: Nova Sonic bridge on Fargate ---
        vpc = ec2.Vpc(self, "TutorVpc", max_azs=2, nat_gateways=1)
        cluster = ecs.Cluster(self, "TutorCluster", vpc=vpc,
                              container_insights=True)

        bridge_image = ecr_assets.DockerImageAsset(self, "BridgeImage",
            directory=os.path.join(ROOT, "bridge"),
            platform=ecr_assets.Platform.LINUX_AMD64)

        service_kwargs = {}
        if certificate_arn:
            service_kwargs.update(
                certificate=acm.Certificate.from_certificate_arn(
                    self, "BridgeCert", certificate_arn),
                protocol=elbv2_protocol_https(),
                redirect_http=True,
            )

        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "SonicBridge",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            public_load_balancer=True,
            # Conversations are long-lived WebSockets; the 60s ALB default would drop
            # the socket during any quiet stretch.
            idle_timeout=Duration.seconds(3600),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(bridge_image),
                container_port=8080,
                environment={
                    "NOVA_SONIC_MODEL_ID": NOVA_SONIC_MODEL_ID,
                    "NOVA_SONIC_VOICE": profile["voiceId"],
                    "BEDROCK_REGION": NOVA_SONIC_REGION,
                    "COACH_MODEL_ID": COACH_MODEL_ID,
                    "KNOWLEDGE_BASE_ID": "SET_VIA_SETUP_KB",
                    "MATERIALS_BUCKET": materials_bucket.bucket_name,
                    "SAGEMAKER_ENDPOINT": endpoint_name,
                    "SYSTEM_PROMPT": conversation_prompt,
                    "COACH_PROMPT": coach_prompt,
                    # All languages, so the client can switch mid-session.
                    "LANGUAGE_BUNDLES": packed,
                    "DEFAULT_LANGUAGE": language,
                    # Learners pause mid-sentence to think. LOW makes Sonic wait, so
                    # the coach receives whole utterances instead of fragments.
                    "ENDPOINTING_SENSITIVITY": "LOW",
                },
                secrets={
                    "ACCESS_TOKEN": ecs.Secret.from_secrets_manager(access_token),
                },
            ),
            **service_kwargs,
        )

        service.target_group.configure_health_check(
            path="/health", healthy_http_codes="200",
            interval=Duration.seconds(30), timeout=Duration.seconds(5))
        # Long-lived sockets shouldn't be torn down abruptly on deploy.
        service.target_group.set_attribute("deregistration_delay.timeout_seconds", "30")

        task_role = service.task_definition.task_role
        # Both actions, deliberately: the bidirectional stream is authorized as
        # InvokeModelWithBidirectionalStream, but bedrock also evaluates InvokeModel on
        # the same model for this API. Granting only the streaming action produces an
        # AccessDeniedException that the SDK does not surface until stream teardown, so
        # the session simply goes silent — an extremely confusing failure to debug.
        task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModelWithBidirectionalStream",
                     "bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{NOVA_SONIC_REGION}::foundation-model/{NOVA_SONIC_MODEL_ID}",
                f"arn:aws:bedrock:{NOVA_SONIC_REGION}::foundation-model/amazon.nova-sonic-v1:0",
            ]))
        # Coach model: Claude Haiku via the cross-region inference profile.
        task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[
                "arn:aws:bedrock:*::foundation-model/anthropic.*",
                f"arn:aws:bedrock:*:{self.account}:inference-profile/us.anthropic.*",
            ]))
        task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["bedrock:Retrieve"],
            resources=[f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/*"]))
        if phoneme_endpoint_arn:
            task_role.add_to_principal_policy(iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint"],
                resources=[phoneme_endpoint_arn]))
        materials_bucket.grant_read(task_role)

        # One conversation pins one WebSocket to one task, so concurrency tracks CPU.
        scaling = service.service.auto_scale_task_count(min_capacity=1, max_capacity=5)
        scaling.scale_on_cpu_utilization("CpuScaling", target_utilization_percent=65)

        # --- Hosting: CloudFront in front of both the page and the WebSocket ---
        #
        # The browser client and the bridge are served from ONE origin on purpose. A
        # page loaded over https cannot open a ws:// socket (mixed content is blocked),
        # and the ALB has no certificate unless you bring one. Routing /ws through the
        # same distribution means CloudFront terminates TLS with its own
        # *.cloudfront.net certificate, so the client talks wss:// to its own origin —
        # no ACM certificate, no custom domain, and no mixed-content problem.
        site_bucket = s3.Bucket(self, "SiteBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        distribution = cloudfront.Distribution(self, "SiteDistribution",
            comment="Language Tutor client + bridge WebSocket",
            default_root_object="index.html",
            # Static client. The bucket stays private; CloudFront reads it through
            # Origin Access Control.
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            additional_behaviors={
                # WebSocket upgrade to the Fargate bridge. CloudFront proxies
                # WebSockets natively, but only if nothing is cached and the upgrade
                # headers (plus the ?token= query string) reach the origin — hence
                # CACHING_DISABLED and the all-viewer origin request policy.
                "/ws": cloudfront.BehaviorOptions(
                    origin=origins.LoadBalancerV2Origin(
                        service.load_balancer,
                        protocol_policy=(
                            cloudfront.OriginProtocolPolicy.HTTPS_ONLY if certificate_arn
                            else cloudfront.OriginProtocolPolicy.HTTP_ONLY),
                        read_timeout=Duration.seconds(60),
                        keepalive_timeout=Duration.seconds(60),
                    ),
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=(
                        cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                ),
            },
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
        )

        # Publish frontend/ on every deploy and invalidate the cache, so a client change
        # ships with `./deploy.sh` instead of needing a manual upload.
        #
        # config.json is generated here rather than derived in the browser. The client
        # can fall back to its own origin, which is correct for this topology, but that
        # assumption breaks the moment the bridge lives on a different host than the
        # page (a dedicated WebSocket domain, a split frontend, another region). Making
        # CloudFormation the single source of truth means the client keeps working
        # without a code change, and nobody has to paste a URL.
        s3deploy.BucketDeployment(self, "SiteContent",
            sources=[
                s3deploy.Source.asset(os.path.join(ROOT, "frontend")),
                s3deploy.Source.json_data("config.json", {
                    "wsUrl": f"wss://{distribution.distribution_domain_name}/ws",
                    "defaultLanguage": language,
                    "languages": [{"id": name, "label": bundles[name]["label"]}
                                  for name in sorted(bundles)],
                }),
            ],
            destination_bucket=site_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
            # None of these filenames are content-hashed, so a browser that cached
            # app.js keeps running last week's client even after the CloudFront
            # invalidation. Revalidate on every load: these are a few KB, and silently
            # running stale client code is a genuinely confusing class of bug.
            cache_control=[
                s3deploy.CacheControl.set_public(),
                s3deploy.CacheControl.max_age(Duration.seconds(0)),
                s3deploy.CacheControl.must_revalidate(),
            ])

        # --- Outputs ---
        CfnOutput(self, "AppUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="Open this and paste the access token. Mic works because "
                        "CloudFront serves the page over https.")
        CfnOutput(self, "WsUrl",
            value=f"wss://{distribution.distribution_domain_name}/ws",
            description="Bridge WebSocket through CloudFront. Append "
                        "?token=<access token>. The browser client fills this in "
                        "itself; the CLI client reads this output.")
        CfnOutput(self, "AlbWsUrl",
            value=f"{'wss' if certificate_arn else 'ws'}://"
                  f"{service.load_balancer.load_balancer_dns_name}/ws",
            description="Direct-to-ALB WebSocket, bypassing CloudFront. Still gated by "
                        "the access token. Useful for debugging; not usable from an "
                        "https page when plain ws://.")
        CfnOutput(self, "AccessTokenSecretName", value=access_token.secret_name,
            description="aws secretsmanager get-secret-value --secret-id <this> "
                        "--query SecretString --output text")
        CfnOutput(self, "MaterialsBucketName", value=materials_bucket.bucket_name)
        CfnOutput(self, "SyncFunctionName", value=sync_lambda.function_name,
            description="Force a Canvas sync: aws lambda invoke --function-name "
                        "<this> /dev/null")
        CfnOutput(self, "PhonemeEndpointName",
            value=endpoint_name or "disabled (-c phonemes=false)")
        CfnOutput(self, "BridgeServiceName", value=service.service.service_name)


def elbv2_protocol_https():
    """Imported lazily so the module has no hard dependency when TLS is unused."""
    from aws_cdk import aws_elasticloadbalancingv2 as elbv2
    return elbv2.ApplicationProtocol.HTTPS
