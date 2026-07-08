"""
WebSocket streaming handler for the Spanish tutor.

This is the scalable, web-ready conversation path. It runs behind an API Gateway
WebSocket API so a browser (or any WS client) can receive the reply incrementally
instead of waiting for one buffered HTTP response.

Why WebSocket + Lambda (vs. the HTTP path in handler.py):
  - API Gateway HTTP API + Lambda proxy BUFFERS the whole response; it cannot
    stream. A WebSocket API lets the backend push many messages per turn.
  - Fully serverless: API Gateway + Lambda scale per-connection with no servers to
    manage. One `converse` message fans out into token deltas + audio chunks pushed
    back over the same connection via ApiGatewayManagementApi.post_to_connection.

Per-turn flow (route: "converse"):
  1. Fetch the recorded utterance from S3 (uploaded via the existing GET /upload-url).
  2. Extract phonemes from SageMaker (Wav2Vec2) — needed by the tutor prompt.
  3. Stream the Bedrock reply token-by-token. As <reply> text arrives it is:
       - pushed to the client as `reply_delta` messages (live text), and
       - chunked per sentence to Polly, whose MP3 is pushed as `audio` messages
         so playback can start before the full reply is generated.
  4. Pronunciation / grammar notes are parsed after the stream and pushed as
     `pronunciation_note` / `grammar_note` (async text annotations).

Client -> server (body, routed on $request.body.action):
    {"action": "converse", "text": "...", "audio_key": "audio/x.wav", "history": [...]}

Server -> client (JSON, one per post_to_connection):
    {"type": "status", "stage": "phonemes" | "thinking"}
    {"type": "reply_delta", "text": "..."}
    {"type": "audio", "seq": 0, "format": "mp3", "data": "<base64>"}
    {"type": "pronunciation_note", "text": "..."}
    {"type": "grammar_note", "text": "..."}
    {"type": "done"}
    {"type": "error", "message": "..."}
"""

import base64
import json
import os
import re

import boto3

SAGEMAKER_ENDPOINT = os.environ["SAGEMAKER_ENDPOINT"]
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
SYSTEM_PROMPT = os.environ["SYSTEM_PROMPT"]
MATERIALS_BUCKET = os.environ["MATERIALS_BUCKET"]

POLLY_VOICE = os.environ.get("POLLY_VOICE", "Lupe")
POLLY_LANG = os.environ.get("POLLY_LANGUAGE_CODE", "es-US")

sagemaker_runtime = boto3.client("sagemaker-runtime")
bedrock = boto3.client("bedrock-runtime")
bedrock_agent = boto3.client("bedrock-agent-runtime")
polly = boto3.client("polly")
s3 = boto3.client("s3")

TOOLS = [{
    "name": "search_materials",
    "description": "Search course materials, vocab lists, lesson plans, assignments, or syllabus. Use for any class-related question.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for"}},
        "required": ["query"],
    },
}]

# Split reply text into speakable chunks on sentence boundaries so Polly can start
# synthesizing before the model finishes. Spanish punctuation included.
_SENTENCE_RE = re.compile(r"[^.!?…\n]*[.!?…\n]+")
_CLOSE_TAG = "</reply>"


def retrieve_from_kb(query):
    if not KNOWLEDGE_BASE_ID or KNOWLEDGE_BASE_ID == "SET_VIA_SETUP_KB":
        return "Knowledge base not configured."
    try:
        resp = bedrock_agent.retrieve(knowledgeBaseId=KNOWLEDGE_BASE_ID, retrievalQuery={"text": query})
        results = resp.get("retrievalResults", [])
        return "\n\n".join(r["content"]["text"] for r in results) if results else "No materials found."
    except Exception as e:
        return f"Error: {e}"


class ReplyStreamer:
    """
    Consumes raw model text deltas, extracts the text inside <reply>...</reply>,
    emits it live to the client, and flushes complete sentences to Polly.

    The full raw text (including the note tags) is retained in `buf` so the caller
    can parse <pronunciation_note>/<grammar_note> after the stream completes.
    """

    def __init__(self, send):
        self.send = send
        self.buf = ""
        self.reply_started = False
        self.reply_ended = False
        self.content_start = 0
        self.emitted = 0
        self.tts_pending = ""
        self.reply_text = ""
        self.seq = 0

    def feed(self, text):
        self.buf += text
        if not self.reply_started:
            i = self.buf.find("<reply>")
            if i == -1:
                return
            self.reply_started = True
            self.content_start = i + len("<reply>")
        if self.reply_ended:
            return

        content = self.buf[self.content_start:]
        end = content.find(_CLOSE_TAG)
        if end != -1:
            available = content[:end]
            self.reply_ended = True
        else:
            # Hold back the tail so we never emit a half-formed "</reply>" tag.
            available = content[:-len(_CLOSE_TAG)] if len(content) > len(_CLOSE_TAG) else ""

        new = available[self.emitted:]
        if new:
            self.emitted += len(new)
            self.reply_text += new
            self.tts_pending += new
            self.send({"type": "reply_delta", "text": new})
            self._flush(final=False)
        if self.reply_ended:
            self._flush(final=True)

    def _flush(self, final):
        while True:
            m = _SENTENCE_RE.match(self.tts_pending) or _SENTENCE_RE.search(self.tts_pending)
            if not m or m.start() != 0:
                break
            sentence = self.tts_pending[:m.end()]
            self.tts_pending = self.tts_pending[m.end():]
            self._tts(sentence.strip())
        if final and self.tts_pending.strip():
            self._tts(self.tts_pending.strip())
            self.tts_pending = ""

    def _tts(self, text):
        if not text:
            return
        try:
            r = polly.synthesize_speech(
                Text=text, OutputFormat="mp3",
                VoiceId=POLLY_VOICE, Engine="neural", LanguageCode=POLLY_LANG)
            audio = r["AudioStream"].read()
            self.send({
                "type": "audio", "seq": self.seq, "format": "mp3",
                "data": base64.b64encode(audio).decode(),
            })
            self.seq += 1
        except Exception as e:
            self.send({"type": "error", "message": f"tts: {e}"})

    def finalize_fallback(self):
        """If the model returned no <reply> tag, treat all text as the reply."""
        if not self.reply_started and self.buf.strip():
            txt = self.buf.strip()
            self.reply_text = txt
            self.tts_pending = txt
            self.send({"type": "reply_delta", "text": txt})
            self._flush(final=True)


def _stream_once(messages, streamer):
    """
    Run one streaming Bedrock call. Feeds text deltas to `streamer`.
    Returns a tool-use request dict {id,name,input} if the model asked for a tool,
    else None.
    """
    resp = bedrock.invoke_model_with_response_stream(
        modelId=BEDROCK_MODEL_ID, contentType="application/json", accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 800,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": TOOLS,
        }))

    tool = None
    cur = None
    for event in resp["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        ctype = chunk.get("type")
        if ctype == "content_block_start":
            block = chunk.get("content_block", {})
            if block.get("type") == "tool_use":
                cur = {"id": block["id"], "name": block["name"], "json": ""}
        elif ctype == "content_block_delta":
            delta = chunk.get("delta", {})
            dt = delta.get("type")
            if dt == "text_delta":
                streamer.feed(delta.get("text", ""))
            elif dt == "input_json_delta" and cur is not None:
                cur["json"] += delta.get("partial_json", "")
        elif ctype == "content_block_stop":
            if cur is not None:
                try:
                    parsed = json.loads(cur["json"] or "{}")
                except json.JSONDecodeError:
                    parsed = {}
                tool = {"id": cur["id"], "name": cur["name"], "input": parsed}
                cur = None
    return tool


def converse_stream(send, expected_phonemes, actual_phonemes, text, history):
    user_msg = (f'[The learner said: "{text}"]\n'
                f'[Expected phonemes: {expected_phonemes}]\n'
                f'[Actual phonemes: {actual_phonemes}]')
    messages = list(history) + [{"role": "user", "content": user_msg}]

    streamer = ReplyStreamer(send)
    send({"type": "status", "stage": "thinking"})

    tool = _stream_once(messages, streamer)
    if tool:
        # Resolve the KB lookup, then continue with a second streaming call that
        # produces the actual reply text.
        kb_results = retrieve_from_kb(tool["input"].get("query", ""))
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tool["id"], "name": tool["name"], "input": tool["input"]}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool["id"], "content": kb_results or "No materials found."}]})
        _stream_once(messages, streamer)

    streamer.finalize_fallback()

    note = re.search(r"<pronunciation_note>(.*?)</pronunciation_note>", streamer.buf, re.DOTALL)
    grammar = re.search(r"<grammar_note>(.*?)</grammar_note>", streamer.buf, re.DOTALL)
    if note and note.group(1).strip():
        send({"type": "pronunciation_note", "text": note.group(1).strip()})
    if grammar and grammar.group(1).strip():
        send({"type": "grammar_note", "text": grammar.group(1).strip()})

    return streamer.reply_text


def _get_phonemes(audio_key, text):
    obj = s3.get_object(Bucket=MATERIALS_BUCKET, Key=audio_key)
    audio_b64 = base64.b64encode(obj["Body"].read()).decode()
    sm_resp = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT, ContentType="application/json",
        Body=json.dumps({"audio_b64": audio_b64, "text": text}))
    result = json.loads(sm_resp["Body"].read())
    return result.get("expected_phonemes", []), result.get("actual_phonemes", [])


def _make_sender(event):
    rc = event["requestContext"]
    endpoint = f"https://{rc['domainName']}/{rc['stage']}"
    conn_id = rc["connectionId"]
    api = boto3.client("apigatewaymanagementapi", endpoint_url=endpoint)

    def send(obj):
        api.post_to_connection(ConnectionId=conn_id, Data=json.dumps(obj).encode())

    return send


def lambda_handler(event, context):
    rc = event.get("requestContext", {})
    route = rc.get("routeKey")

    # Connection lifecycle routes: nothing to do, but must return 200.
    if route in ("$connect", "$disconnect"):
        return {"statusCode": 200}

    send = _make_sender(event)
    try:
        body = json.loads(event.get("body") or "{}")
        text = body.get("text", "")
        audio_key = body.get("audio_key", "")
        history = body.get("history", [])

        if not text or not audio_key:
            send({"type": "error", "message": "Missing text or audio_key"})
            return {"statusCode": 200}

        send({"type": "status", "stage": "phonemes"})
        expected, actual = _get_phonemes(audio_key, text)

        converse_stream(send, expected, actual, text, history)
        send({"type": "done"})
    except Exception as e:
        try:
            send({"type": "error", "message": str(e)})
        except Exception:
            pass
    return {"statusCode": 200}
