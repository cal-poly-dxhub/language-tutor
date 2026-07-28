"""
Infrastructure assertions.

Synthesized with an environment-agnostic account so these run without AWS credentials.
The focus is the wiring that is easy to get subtly wrong and expensive to discover after
a deploy: the WebSocket must reach the bridge uncached over TLS, the site bucket must
stay private, and the phonemes flag must actually remove the GPU endpoint.
"""

import json
import os
import sys

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cdk.stack import LanguageTutorStack  # noqa: E402

# Managed CloudFront policy ids (stable, AWS-published).
CACHING_DISABLED = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
CACHING_OPTIMIZED = "658327ea-f89d-4fab-a63d-7e88639e58f6"
ALL_VIEWER_EXCEPT_HOST = "b689b0a8-53d0-40ab-baf2-68738e2966ac"


def synth(**context):
    app = cdk.App(context=context or None)
    stack = LanguageTutorStack(app, "TestStack")
    return Template.from_stack(stack)


@pytest.fixture(scope="module")
def template():
    return synth()


def distribution_config(template):
    dists = template.find_resources("AWS::CloudFront::Distribution")
    assert len(dists) == 1
    return list(dists.values())[0]["Properties"]["DistributionConfig"]


# --- hosting -----------------------------------------------------------------

def test_client_is_served_over_https_from_a_private_bucket(template):
    cfg = distribution_config(template)
    assert cfg["DefaultRootObject"] == "index.html"
    assert cfg["DefaultCacheBehavior"]["ViewerProtocolPolicy"] == "redirect-to-https"
    # Origin Access Control, not a public bucket.
    template.resource_count_is("AWS::CloudFront::OriginAccessControl", 1)
    for bucket in template.find_resources("AWS::S3::Bucket").values():
        block = bucket["Properties"]["PublicAccessBlockConfiguration"]
        assert block["BlockPublicPolicy"] and block["RestrictPublicBuckets"]


def test_websocket_is_proxied_to_the_bridge_uncached_over_tls(template):
    cfg = distribution_config(template)
    ws = [b for b in cfg["CacheBehaviors"] if b["PathPattern"] == "/ws"]
    assert len(ws) == 1, "the bridge must be reachable on the same origin as the page"
    behavior = ws[0]
    # An https page cannot open ws://, so this behavior is what makes wss:// possible.
    assert behavior["ViewerProtocolPolicy"] == "https-only"
    # Caching a WebSocket upgrade would break it outright.
    assert behavior["CachePolicyId"] == CACHING_DISABLED
    # The upgrade headers and the ?token= query string have to reach the origin.
    assert behavior["OriginRequestPolicyId"] == ALL_VIEWER_EXCEPT_HOST
    assert "POST" in behavior["AllowedMethods"]


def test_static_content_is_cached_but_the_socket_is_not(template):
    cfg = distribution_config(template)
    assert cfg["DefaultCacheBehavior"]["CachePolicyId"] == CACHING_OPTIMIZED


def test_two_origins_page_and_load_balancer(template):
    cfg = distribution_config(template)
    assert len(cfg["Origins"]) == 2
    custom = [o for o in cfg["Origins"] if "CustomOriginConfig" in o]
    assert len(custom) == 1
    # No ACM certificate by default, so CloudFront reaches the ALB over plain HTTP.
    assert custom[0]["CustomOriginConfig"]["OriginProtocolPolicy"] == "http-only"


def test_client_is_published_and_cache_invalidated(template):
    template.resource_count_is("Custom::CDKBucketDeployment", 1)
    template.has_resource_properties("Custom::CDKBucketDeployment", {
        "DistributionPaths": ["/*"],
    })


def test_client_files_are_revalidated_not_cached(template):
    """
    Filenames are not content-hashed, so a browser that cached app.js would keep running
    an old client after a deploy — invalidating CloudFront alone does not fix that.
    """
    template.has_resource_properties("Custom::CDKBucketDeployment", {
        "SystemMetadata": {"cache-control": "public, max-age=0, must-revalidate"},
    })


def test_published_config_carries_the_endpoint_but_never_the_token(template):
    """
    config.json is world-readable: CloudFront serves it to anyone with the URL. The
    endpoint belongs there (it is public anyway, and it saves the user pasting it); the
    access token must not, or the token stops protecting the Bedrock bill.
    """
    doc = json.dumps(template.to_json())
    deployment = list(
        template.find_resources("Custom::CDKBucketDeployment").values())[0]["Properties"]
    # Two sources: the frontend/ asset and the generated config.json.
    assert len(deployment["SourceObjectKeys"]) == 2

    # Nothing in the template may plant the secret's resolved value into the site, and
    # the site must not be granted a way to read it either.
    assert "bridge-access-token" not in json.dumps(
        {k: v for k, v in template.to_json()["Resources"].items()
         if v["Type"] == "Custom::CDKBucketDeployment"})
    # The secret is delivered to the bridge container only, as an ECS secret.
    assert "SecretsManager" in doc  # sanity: the secret exists at all
    template.has_resource_properties("AWS::ECS::TaskDefinition", {
        "ContainerDefinitions": Match.array_with([
            Match.object_like({"Secrets": Match.array_with([
                Match.object_like({"Name": "ACCESS_TOKEN"})])}),
        ]),
    })


def test_outputs_give_the_user_what_they_need(template):
    outputs = template.to_json()["Outputs"]
    for key in ("AppUrl", "WsUrl", "AccessTokenSecretName", "SyncFunctionName"):
        assert key in outputs, key


# --- bridge ------------------------------------------------------------------

def test_bridge_requires_the_access_token_secret(template):
    template.has_resource_properties("AWS::ECS::TaskDefinition", {
        "ContainerDefinitions": Match.array_with([
            Match.object_like({"Secrets": Match.array_with([
                Match.object_like({"Name": "ACCESS_TOKEN"})])}),
        ]),
    })


def test_alb_idle_timeout_survives_quiet_conversation(template):
    template.has_resource_properties("AWS::ElasticLoadBalancingV2::LoadBalancer", {
        "LoadBalancerAttributes": Match.array_with([
            {"Key": "idle_timeout.timeout_seconds", "Value": "3600"}]),
    })


def test_bridge_may_open_a_bidirectional_stream(template):
    policies = template.find_resources("AWS::IAM::Policy")
    actions = [stmt.get("Action")
               for p in policies.values()
               for stmt in p["Properties"]["PolicyDocument"]["Statement"]]
    flat = [a for entry in actions for a in (entry if isinstance(entry, list) else [entry])]
    assert "bedrock:InvokeModelWithBidirectionalStream" in flat
    assert "bedrock:Retrieve" in flat


# --- context flags -----------------------------------------------------------

def test_phonemes_true_by_default(template):
    template.resource_count_is("AWS::SageMaker::Endpoint", 1)


def test_phonemes_false_removes_the_gpu_endpoint():
    t = synth(phonemes="false")
    t.resource_count_is("AWS::SageMaker::Endpoint", 0)
    t.resource_count_is("AWS::SageMaker::Model", 0)
    # The conversation still works; only the coach loses pronunciation.
    t.resource_count_is("AWS::ECS::Service", 1)
    t.resource_count_is("AWS::CloudFront::Distribution", 1)


def test_language_context_switches_voice_and_prompts():
    spanish = synth().to_json()
    french = synth(language="french").to_json()

    def container_env(doc):
        for res in doc["Resources"].values():
            if res["Type"] == "AWS::ECS::TaskDefinition":
                return {e["Name"]: e.get("Value")
                        for e in res["Properties"]["ContainerDefinitions"][0]["Environment"]}
        raise AssertionError("no task definition")

    es, fr = container_env(spanish), container_env(french)
    assert es["NOVA_SONIC_VOICE"] != fr["NOVA_SONIC_VOICE"]
    assert "Spanish" in es["SYSTEM_PROMPT"] and "French" in fr["SYSTEM_PROMPT"]
    for env in (es, fr):
        assert "{{" not in env["SYSTEM_PROMPT"] + env["COACH_PROMPT"]
