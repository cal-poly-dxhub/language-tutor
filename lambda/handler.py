"""
Lambda handler for the custom-bot mode of the foreign language tutor.

This is the "does everything" pipeline (as opposed to raw Nova Sonic):
    audio (S3) -> Amazon Transcribe (STT) -> SageMaker Wav2Vec2 (phonemes)
               -> Bedrock Claude (conversation + evaluation, + KB tool) -> Polly (TTS)

HTTP API endpoints:
  GET  /upload-url → presigned S3 PUT URL for the recorded utterance
  POST /converse   → {audio_key, history, [text]} → reply + transcript + notes + audio URL

STT lives here (server-side) so the browser only has to record and upload a WAV. If the
caller already has a transcript (e.g. the CLI client that streams Transcribe itself), it
can pass `text` and STT is skipped.
"""

import asyncio
import base64
import json
import os
import re
import uuid
import wave

import boto3

SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
SYSTEM_PROMPT = os.environ["SYSTEM_PROMPT"]
MATERIALS_BUCKET = os.environ["MATERIALS_BUCKET"]
TRANSCRIBE_LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "es-US")
POLLY_VOICE = os.environ.get("POLLY_VOICE", "Lupe")
POLLY_LANGUAGE = os.environ.get("POLLY_LANGUAGE", "es-US")

sagemaker_runtime = boto3.client("sagemaker-runtime")
bedrock = boto3.client("bedrock-runtime")
bedrock_agent = boto3.client("bedrock-agent-runtime")
polly = boto3.client("polly")
s3 = boto3.client("s3")

TOOLS = [{"name": "search_materials", "description": "Search course materials, vocab lists, lesson plans, assignments, or syllabus. Use for any class-related question.", "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "What to search for"}}, "required": ["query"]}}]

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}


# --- Amazon Transcribe streaming (server-side STT) ---------------------------
def transcribe_wav(audio_bytes: bytes) -> str:
    """Transcribe a mono PCM16 WAV using Amazon Transcribe streaming.

    Streaming (not batch) so a short utterance returns text in a couple of seconds,
    well inside the Lambda timeout. The amazon-transcribe SDK is async, so we drive a
    private event loop for the duration of this one call.
    """
    import io

    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent

    region = os.environ.get("AWS_REGION", "us-west-2")

    class _Handler(TranscriptResultStreamHandler):
        def __init__(self, stream):
            super().__init__(stream)
            self.text = ""

        async def handle_transcript_event(self, event: TranscriptEvent):
            for result in event.transcript.results:
                if not result.is_partial:
                    for alt in result.alternatives:
                        self.text += alt.transcript + " "

    async def _run() -> str:
        client = TranscribeStreamingClient(region=region)
        stream = await client.start_stream_transcription(
            language_code=TRANSCRIBE_LANGUAGE,
            media_sample_rate_hz=sample_rate,
            media_encoding="pcm",
        )
        handler = _Handler(stream.output_stream)

        async def feed():
            chunk = 1024 * 8
            for i in range(0, len(pcm), chunk):
                await stream.input_stream.send_audio_event(audio_chunk=pcm[i:i + chunk])
            await stream.input_stream.end_stream()

        await asyncio.gather(feed(), handler.handle_events())
        return handler.text.strip()

    return asyncio.run(_run())


# --- Knowledge Base + LLM ----------------------------------------------------
def retrieve_from_kb(query):
    if not KNOWLEDGE_BASE_ID or KNOWLEDGE_BASE_ID == "SET_VIA_SETUP_KB":
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
    if expected_phonemes or actual_phonemes:
        user_msg = f'[The learner said: "{text}"]\n[Expected phonemes: {expected_phonemes}]\n[Actual phonemes: {actual_phonemes}]'
    else:
        user_msg = f'[The learner said: "{text}"]'
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


# --- HTTP handlers -----------------------------------------------------------
def _json(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json", **CORS},
            "body": json.dumps(body)}


def handle_upload_url(event):
    """Generate presigned URL for audio upload."""
    key = f"audio/{uuid.uuid4()}.wav"
    url = s3.generate_presigned_url("put_object",
        Params={"Bucket": MATERIALS_BUCKET, "Key": key, "ContentType": "audio/wav"},
        ExpiresIn=300)
    return _json(200, {"upload_url": url, "key": key})


def handle_converse(event):
    """Process a conversation turn."""
    body = json.loads(event.get("body", "{}"))
    audio_key = body.get("audio_key", "")
    text = (body.get("text") or "").strip()
    history = body.get("history", [])

    if not audio_key:
        return _json(400, {"error": "Missing audio_key"})

    # Get audio from S3
    obj = s3.get_object(Bucket=MATERIALS_BUCKET, Key=audio_key)
    audio_bytes = obj["Body"].read()

    # Server-side STT when the caller didn't provide a transcript.
    if not text:
        try:
            text = transcribe_wav(audio_bytes)
        except Exception as e:  # noqa: BLE001
            return _json(500, {"error": f"Transcription failed: {e}"})
        if not text:
            return _json(200, {"reply": "", "transcript": "", "error": "no_speech"})

    # Phonemes from SageMaker (optional — skipped if no endpoint is configured).
    actual, expected = [], []
    if SAGEMAKER_ENDPOINT:
        try:
            audio_b64 = base64.b64encode(audio_bytes).decode()
            sm_resp = sagemaker_runtime.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT, ContentType="application/json",
                Body=json.dumps({"audio_b64": audio_b64, "text": text}))
            phoneme_result = json.loads(sm_resp["Body"].read())
            actual = phoneme_result.get("actual_phonemes", [])
            expected = phoneme_result.get("expected_phonemes", [])
        except Exception:  # noqa: BLE001
            pass  # phoneme scoring is best-effort; grammar feedback still works

    # LLM conversation
    llm_result = converse(expected, actual, text, history)

    # Polly TTS → S3
    polly_resp = polly.synthesize_speech(
        Text=llm_result["reply"], OutputFormat="mp3", VoiceId=POLLY_VOICE,
        Engine="neural", LanguageCode=POLLY_LANGUAGE)
    reply_key = f"audio/reply-{uuid.uuid4()}.mp3"
    s3.put_object(Bucket=MATERIALS_BUCKET, Key=reply_key, Body=polly_resp["AudioStream"].read())
    reply_url = s3.generate_presigned_url("get_object",
        Params={"Bucket": MATERIALS_BUCKET, "Key": reply_key}, ExpiresIn=300)

    return _json(200, {
        "transcript": text,
        "reply": llm_result["reply"],
        "pronunciation_note": llm_result.get("pronunciation_note"),
        "grammar_note": llm_result.get("grammar_note"),
        "reply_audio_url": reply_url,
    })


def lambda_handler(event, context):
    # HTTP API routing
    if "routeKey" in event:
        route = event["routeKey"]
        if route.startswith("OPTIONS"):
            return {"statusCode": 204, "headers": CORS, "body": ""}
        if route == "GET /upload-url":
            return handle_upload_url(event)
        if route == "POST /converse":
            return handle_converse(event)

    return {"statusCode": 404, "headers": CORS, "body": "Not found"}
