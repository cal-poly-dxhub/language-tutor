"""CDK stack for the foreign language tutor conversational bot."""

import os
from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_ecr_assets as ecr_assets,
    aws_lambda as _lambda,
    aws_apigatewayv2 as apigwv2,
    aws_events as events,
    aws_events_targets as targets,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class PronunciationCheckerStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # --- S3 Buckets ---
        model_bucket = s3.Bucket(self, "ModelArtifacts",
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        materials_bucket = s3.Bucket(self, "MaterialsBucket",
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        # --- SageMaker (phoneme extraction) ---
        sagemaker_role = iam.Role(self, "SageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")])
        model_bucket.grant_read(sagemaker_role)

        model_image = ecr_assets.DockerImageAsset(self, "Wav2Vec2Image",
            directory=os.path.join(os.path.dirname(__file__), "..", "container"),
            platform=ecr_assets.Platform.LINUX_AMD64)

        model = sagemaker.CfnModel(self, "PhonemeModel",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=model_image.image_uri, mode="SingleModel"))
        model_image.repository.grant_pull(sagemaker_role)

        endpoint_config = sagemaker.CfnEndpointConfig(self, "EndpointConfig",
            production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                variant_name="AllTraffic", model_name=model.attr_model_name,
                initial_instance_count=1, instance_type="ml.g4dn.xlarge", initial_variant_weight=1.0)])
        endpoint_config.add_dependency(model)

        endpoint = sagemaker.CfnEndpoint(self, "PhonemeEndpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name)
        endpoint.add_dependency(endpoint_config)

        # --- Canvas API token in Secrets Manager ---
        canvas_secret = secretsmanager.Secret(self, "CanvasApiToken",
            secret_name="tutor/canvas-api-token",
            description="Canvas LMS API token for syncing course materials")

        # --- Canvas sync Lambda ---
        sync_lambda = _lambda.Function(self, "CanvasSyncHandler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="sync.lambda_handler",
            code=_lambda.Code.from_asset(os.path.join(os.path.dirname(__file__), "..", "lambda")),
            timeout=Duration.seconds(300),
            memory_size=512,
            environment={
                "MATERIALS_BUCKET": materials_bucket.bucket_name,
                "CANVAS_SECRET_ARN": canvas_secret.secret_arn,
            })
        materials_bucket.grant_write(sync_lambda)
        canvas_secret.grant_read(sync_lambda)
        sync_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:StartIngestionJob", "bedrock:ListDataSources", "bedrock:ListKnowledgeBases"],
            resources=["*"]))

        # --- EventBridge: sync every 6 hours ---
        events.Rule(self, "SyncSchedule",
            schedule=events.Schedule.rate(Duration.hours(6)),
            targets=[targets.LambdaFunction(sync_lambda)])

        # --- HTTP API for force sync + conversation ---
        http_api = apigwv2.CfnApi(self, "SyncApi",
            name="TutorApi", protocol_type="HTTP")

        sync_integration = apigwv2.CfnIntegration(self, "SyncIntegration",
            api_id=http_api.ref, integration_type="AWS_PROXY", integration_method="POST",
            integration_uri=sync_lambda.function_arn, payload_format_version="2.0")

        apigwv2.CfnRoute(self, "SyncRoute",
            api_id=http_api.ref, route_key="POST /sync",
            target=f"integrations/{sync_integration.ref}")

        apigwv2.CfnStage(self, "SyncStage",
            api_id=http_api.ref, stage_name="$default", auto_deploy=True)

        sync_lambda.add_permission("HttpApiInvoke",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=f"arn:aws:execute-api:{self.region}:{self.account}:{http_api.ref}/*")

        # --- Main conversation Lambda ---
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "system.txt")
        with open(prompt_path) as f:
            system_prompt = f.read()

        lambda_role = iam.Role(self, "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")])

        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream"],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{endpoint.attr_endpoint_name}"]))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[
                "arn:aws:bedrock:*::foundation-model/anthropic.*",
                f"arn:aws:bedrock:*:{self.account}:inference-profile/us.anthropic.*",
            ]))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:Retrieve"],
            resources=[f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/*"]))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["polly:SynthesizeSpeech"], resources=["*"]))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject"],
            resources=[f"{materials_bucket.bucket_arn}/*"]))

        handler = _lambda.Function(self, "PronunciationHandler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset(os.path.join(os.path.dirname(__file__), "..", "lambda")),
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=512,
            environment={
                "SAGEMAKER_ENDPOINT": endpoint.attr_endpoint_name,
                "SYSTEM_PROMPT": system_prompt,
                "KNOWLEDGE_BASE_ID": "SET_VIA_SETUP_KB",
                "MATERIALS_BUCKET": materials_bucket.bucket_name,
            })

        # --- HTTP API converse routes (added after handler defined) ---
        converse_integration = apigwv2.CfnIntegration(self, "ConverseIntegration",
            api_id=http_api.ref, integration_type="AWS_PROXY", integration_method="POST",
            integration_uri=handler.function_arn, payload_format_version="2.0")

        apigwv2.CfnRoute(self, "ConverseRoute",
            api_id=http_api.ref, route_key="POST /converse",
            target=f"integrations/{converse_integration.ref}")

        apigwv2.CfnRoute(self, "UploadRoute",
            api_id=http_api.ref, route_key="GET /upload-url",
            target=f"integrations/{converse_integration.ref}")

        handler.add_permission("HttpApiConverseInvoke",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=f"arn:aws:execute-api:{self.region}:{self.account}:{http_api.ref}/*")

        # --- Outputs ---
        CfnOutput(self, "ApiUrl",
            value=f"https://{http_api.ref}.execute-api.{self.region}.amazonaws.com")
        CfnOutput(self, "SyncUrl",
            value=f"https://{http_api.ref}.execute-api.{self.region}.amazonaws.com/sync")
        CfnOutput(self, "MaterialsBucketName", value=materials_bucket.bucket_name)
        CfnOutput(self, "EndpointName", value=endpoint.attr_endpoint_name)
