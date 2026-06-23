"""
Lambda handler for conversational Spanish pronunciation bot.

The bot has a conversation in Spanish with the learner. It responds naturally,
and notes pronunciation issues in a separate field (not the main conversation).
"""

import base64
import json
import os

import boto3

SAGEMAKER_ENDPOINT = os.environ["SAGEMAKER_ENDPOINT"]
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

sagemaker_runtime = boto3.client("sagemaker-runtime")
bedrock = boto3.client("bedrock-runtime")
polly = boto3.client("polly")

SYSTEM_PROMPT = os.environ["SYSTEM_PROMPT"]


def post_to_connection(event, data):
    domain = event["requestContext"]["domainName"]
    stage = event["requestContext"]["stage"]
    connection_id = event["requestContext"]["connectionId"]
    apigw = boto3.client("apigatewaymanagementapi", endpoint_url=f"https://{domain}/{stage}")
    apigw.post_to_connection(ConnectionId=connection_id, Data=json.dumps(data).encode())


def converse(expected_phonemes, actual_phonemes, text, history):
    """LLM generates a conversational reply + pronunciation notes."""
    messages = history + [{"role": "user", "content": f'[The learner said: "{text}"]\n[Expected phonemes: {expected_phonemes}]\n[Actual phonemes: {actual_phonemes}]'}]

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }),
    )
    result = json.loads(response["body"].read())
    import re
    text_out = result["content"][0]["text"]
    reply = re.search(r'<reply>(.*?)</reply>', text_out, re.DOTALL)
    note = re.search(r'<pronunciation_note>(.*?)</pronunciation_note>', text_out, re.DOTALL)
    grammar = re.search(r'<grammar_note>(.*?)</grammar_note>', text_out, re.DOTALL)
    return {
        "reply": reply.group(1).strip() if reply else text_out,
        "pronunciation_note": note.group(1).strip() if note and note.group(1).strip() else None,
        "grammar_note": grammar.group(1).strip() if grammar and grammar.group(1).strip() else None,
    }


def synthesize_speech(text):
    """Generate Spanish speech via Polly."""
    response = polly.synthesize_speech(
        Text=text, OutputFormat="mp3", VoiceId="Lupe", Engine="neural", LanguageCode="es-US"
    )
    return base64.b64encode(response["AudioStream"].read()).decode()


def lambda_handler(event, context):
    route = event["requestContext"].get("routeKey", "$default")

    if route in ("$connect", "$disconnect"):
        return {"statusCode": 200}

    body = json.loads(event.get("body", "{}"))

    if body.get("action") != "check":
        return {"statusCode": 200}

    audio_b64 = body.get("audio", "")
    text = body.get("text", "")
    history = body.get("history", [])

    if not audio_b64 or not text:
        post_to_connection(event, {"error": "Missing audio or text"})
        return {"statusCode": 200}

    try:
        # Extract phonemes via SageMaker
        sm_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"audio_b64": audio_b64, "text": text}),
        )
        result = json.loads(sm_response["Body"].read())
        actual_phonemes = result.get("actual_phonemes", [])
        expected_phonemes = result.get("expected_phonemes", [])

        # Conversational LLM response + pronunciation eval
        llm_result = converse(expected_phonemes, actual_phonemes, text, history)

        # Speak the reply in Spanish via Polly
        reply_audio_b64 = synthesize_speech(llm_result["reply"])

        post_to_connection(event, {
            "text": text,
            "expected_phonemes": expected_phonemes,
            "actual_phonemes": actual_phonemes,
            "reply": llm_result["reply"],
            "pronunciation_note": llm_result.get("pronunciation_note"),
            "score": llm_result.get("score"),
            "reply_audio_b64": reply_audio_b64,
        })
    except Exception as e:
        post_to_connection(event, {"error": str(e)})

    return {"statusCode": 200}
