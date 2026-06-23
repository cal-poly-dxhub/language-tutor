"""CDK stack for the pronunciation checker conversational bot."""

import os
from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_ecr_assets as ecr_assets,
    aws_lambda as _lambda,
    aws_apigatewayv2 as apigwv2,
)
from constructs import Construct


class PronunciationCheckerStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # S3 bucket for model artifacts
        model_bucket = s3.Bucket(self, "ModelArtifacts",
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        # SageMaker execution role
        sagemaker_role = iam.Role(self, "SageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")])
        model_bucket.grant_read(sagemaker_role)

        # Docker image for Wav2Vec2 phoneme model
        model_image = ecr_assets.DockerImageAsset(self, "Wav2Vec2Image",
            directory=os.path.join(os.path.dirname(__file__), "..", "container"),
            platform=ecr_assets.Platform.LINUX_AMD64)

        # SageMaker Model
        model = sagemaker.CfnModel(self, "PhonemeModel",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=model_image.image_uri, mode="SingleModel"))
        model_image.repository.grant_pull(sagemaker_role)

        # Endpoint configuration
        endpoint_config = sagemaker.CfnEndpointConfig(self, "EndpointConfig",
            production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                variant_name="AllTraffic",
                model_name=model.attr_model_name,
                initial_instance_count=1,
                instance_type="ml.g4dn.xlarge",
                initial_variant_weight=1.0)])
        endpoint_config.add_dependency(model)

        # SageMaker Endpoint
        endpoint = sagemaker.CfnEndpoint(self, "PhonemeEndpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name)
        endpoint.add_dependency(endpoint_config)

        # Lambda role
        lambda_role = iam.Role(self, "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")])

        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream"],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{endpoint.attr_endpoint_name}"]))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.*"]))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["polly:SynthesizeSpeech"], resources=["*"]))

        # Read system prompt to pass as env var
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "system.txt")
        with open(prompt_path) as f:
            system_prompt = f.read()

        # Lambda function
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
            })

        # WebSocket API
        ws_api = apigwv2.CfnApi(self, "WebSocketApi",
            name="PronunciationWebSocket",
            protocol_type="WEBSOCKET",
            route_selection_expression="$request.body.action")

        integration = apigwv2.CfnIntegration(self, "LambdaIntegration",
            api_id=ws_api.ref,
            integration_type="AWS_PROXY",
            integration_uri=f"arn:aws:apigateway:{self.region}:lambda:path/2015-03-31/functions/{handler.function_arn}/invocations")

        for route_key in ["$connect", "$disconnect", "$default", "check"]:
            apigwv2.CfnRoute(self, f"Route_{route_key.replace('$', '')}",
                api_id=ws_api.ref, route_key=route_key,
                target=f"integrations/{integration.ref}")

        apigwv2.CfnStage(self, "ProdStage",
            api_id=ws_api.ref, stage_name="prod", auto_deploy=True)

        handler.add_permission("ApiGwInvoke",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=f"arn:aws:execute-api:{self.region}:{self.account}:{ws_api.ref}/*")

        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["execute-api:ManageConnections"],
            resources=[f"arn:aws:execute-api:{self.region}:{self.account}:{ws_api.ref}/prod/*"]))

        # Outputs
        CfnOutput(self, "WebSocketUrl",
            value=f"wss://{ws_api.ref}.execute-api.{self.region}.amazonaws.com/prod")
        CfnOutput(self, "ModelBucket", value=model_bucket.bucket_name)
        CfnOutput(self, "EndpointName", value=endpoint.attr_endpoint_name)
