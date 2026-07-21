#!/usr/bin/env python3
import os
import aws_cdk as cdk
from cdk.stack import PronunciationCheckerStack
from cdk.nova_sonic_stack import NovaSonicTutorStack, NOVA_SONIC_REGION

app = cdk.App()

# Original pipeline: Transcribe -> Wav2Vec2 (SageMaker) -> Claude -> Polly.
PronunciationCheckerStack(app, "PronunciationCheckerStack", env=cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
))

# Nova Sonic variant: single speech-to-speech model behind a Fargate WebSocket bridge.
# Deployed in a Nova Sonic region (us-east-1 by default) so the model, Knowledge Base,
# and retrieve calls all live in the same region.
NovaSonicTutorStack(app, "NovaSonicTutorStack", env=cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("NOVA_SONIC_CDK_REGION", NOVA_SONIC_REGION),
))

app.synth()
