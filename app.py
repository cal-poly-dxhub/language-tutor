#!/usr/bin/env python3
import os
import aws_cdk as cdk
from cdk.stack import LanguageTutorStack

app = cdk.App()
LanguageTutorStack(app, "LanguageTutorStack", env=cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
))
app.synth()
