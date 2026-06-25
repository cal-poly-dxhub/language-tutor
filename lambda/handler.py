"""
Lambda handler for foreign language tutor.

HTTP API endpoints:
  GET /upload-url → returns presigned S3 upload URL for audio
  POST /converse  → processes audio from S3, returns LLM response + audio URL

Also serves WebSocket for programmatic clients.
"""

import base64
import json
import os
import re
import uuid

import boto3

SAGEMAKER_ENDPOINT = os.environ["SAGEMAKER_ENDPOINT"]
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
SYSTEM_PROMPT = os.environ["SYSTEM_PROMPT"]
MATERIALS_BUCKET = os.environ["MATERIALS_BUCKET"]

sagemaker_runtime = boto3.client("sagemaker-runtime")
bedrock = boto3.client("bedrock-runtime")
bedrock_agent = boto3.client("bedrock-agent-runtime")
polly = boto3.client("polly")
s3 = boto3.client("s3")

TOOLS = [{"name": "search_materials", "description": "Search course materials, vocab lists, lesson plans, assignments, or syllabus. Use for any class-related question.", "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "What to search for"}}, "required": ["query"]}}]


def retrieve_from_kb(query):
    if not KNOWLEDGE_BASE_ID:
        return "Knowledge base not configured."
    try:
        resp = bedrock_agent.retrieve(knowledgeBaseId=KNOWLEDGE_BASE_ID, retrievalQuery={"text": query})
        results = resp.get("retrievalResults", [])
        return "\n\n".join(r["content"]["text"] for r in results) if results else "No materials found."
    except Exception as e:
        return f"Error: {e}"


def invoke_llm(messages):
    return json.loads(bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID, contentType="application/json", accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 800,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": TOOLS,
        })).get("body").read())


def converse(expected_phonemes, actual_phonemes, text, history):
    user_msg = f'[The learner said: "{text}"]\n[Expected phonemes: {expected_phonemes}]\n[Actual phonemes: {actual_phonemes}]'
    messages = history + [{"role": "user", "content": user_msg}]

    result = invoke_llm(messages)

    # Handle tool use
    tool_blocks = [b for b in result["content"] if b.get("type") == "tool_use"]
    if tool_blocks:
        tool_block = tool_blocks[0]
        kb_results = retrieve_from_kb(tool_block["input"]["query"])
        messages.append({"role": "assistant", "content": result["content"]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block["id"], "content": kb_results or "No materials found."}]})
        result = invoke_llm(messages)

    text_out = next((b["text"] for b in result["content"] if b.get("type") == "text"), "")
    reply = re.search(r'<reply>(.*?)</reply>', text_out, re.DOTALL)
    note = re.search(r'<pronunciation_note>(.*?)</pronunciation_note>', text_out, re.DOTALL)
    grammar = re.search(r'<grammar_note>(.*?)</grammar_note>', text_out, re.DOTALL)
    return {
        "reply": reply.group(1).strip() if reply else text_out,
        "pronunciation_note": note.group(1).strip() if note and note.group(1).strip() else None,
        "grammar_note": grammar.group(1).strip() if grammar and grammar.group(1).strip() else None,
    }


def handle_upload_url(event):
    """Generate presigned URL for audio upload."""
    key = f"audio/{uuid.uuid4()}.wav"
    url = s3.generate_presigned_url("put_object",
        Params={"Bucket": MATERIALS_BUCKET, "Key": key, "ContentType": "audio/wav"},
        ExpiresIn=300)
    return {"statusCode": 200, "body": json.dumps({"upload_url": url, "key": key}),
            "headers": {"Content-Type": "application/json"}}


def handle_converse(event):
    """Process a conversation turn."""
    body = json.loads(event.get("body", "{}"))
    audio_key = body.get("audio_key", "")
    text = body.get("text", "")
    history = body.get("history", [])

    if not audio_key or not text:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing audio_key or text"})}

    # Get audio from S3
    obj = s3.get_object(Bucket=MATERIALS_BUCKET, Key=audio_key)
    audio_b64 = base64.b64encode(obj["Body"].read()).decode()

    # Phonemes from SageMaker
    sm_resp = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT, ContentType="application/json",
        Body=json.dumps({"audio_b64": audio_b64, "text": text}))
    phoneme_result = json.loads(sm_resp["Body"].read())
    actual = phoneme_result.get("actual_phonemes", [])
    expected = phoneme_result.get("expected_phonemes", [])

    # LLM conversation
    llm_result = converse(expected, actual, text, history)

    # Polly TTS → S3
    polly_resp = polly.synthesize_speech(
        Text=llm_result["reply"], OutputFormat="mp3", VoiceId="Lupe", Engine="neural", LanguageCode="es-US")
    reply_key = f"audio/reply-{uuid.uuid4()}.mp3"
    s3.put_object(Bucket=MATERIALS_BUCKET, Key=reply_key, Body=polly_resp["AudioStream"].read())
    reply_url = s3.generate_presigned_url("get_object",
        Params={"Bucket": MATERIALS_BUCKET, "Key": reply_key}, ExpiresIn=300)

    return {"statusCode": 200, "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "reply": llm_result["reply"],
                "pronunciation_note": llm_result.get("pronunciation_note"),
                "grammar_note": llm_result.get("grammar_note"),
                "reply_audio_url": reply_url,
            })}


def lambda_handler(event, context):
    # HTTP API routing
    if "routeKey" in event:
        route = event["routeKey"]
        if route == "GET /upload-url":
            return handle_upload_url(event)
        if route == "POST /converse":
            return handle_converse(event)

    return {"statusCode": 404, "body": "Not found"}
