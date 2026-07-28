#!/usr/bin/env python3
"""
CDK entry point.

One stack: `LanguageTutorStack` — Nova 2 Sonic speech-to-speech conversation plus the
asynchronous pronunciation/grammar coach. It deploys to a Nova Sonic region
(us-west-2 by default) so the model, the Knowledge Base and the coach all live together.

Deploy through the wrapper, not `npx cdk` directly — the stack builds container images
and the wrapper points CDK at a builder that works on macOS:

    ./deploy.sh                        # deploy
    ./deploy.sh -c language=french     # teach a different language
    ./deploy.sh -c phonemes=false      # skip the GPU phoneme endpoint
"""

import os

import aws_cdk as cdk

from cdk import preflight
from cdk.stack import NOVA_SONIC_REGION, LanguageTutorStack

# Fails fast with instructions if no image builder is reachable, rather than letting
# `cdk deploy` die on a docker socket error after synth succeeds.
preflight.check()

app = cdk.App()

LanguageTutorStack(app, "LanguageTutorStack", env=cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("LANGUAGE_TUTOR_REGION", NOVA_SONIC_REGION),
))

app.synth()
