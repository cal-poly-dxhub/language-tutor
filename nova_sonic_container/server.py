"""
Nova Sonic bridge server (runs on Fargate).

Bridges a browser WebSocket <-> the Bedrock Nova Sonic bidirectional stream. This is
the piece that makes "real-time streaming speech with pronunciation/grammar feedback
without interrupting the conversation" work with a single speech-to-speech model.

WHY A SERVER (not Lambda)
-------------------------
Nova Sonic's `InvokeModelWithBidirectionalStream` is a persistent HTTP/2 event stream.
Lambda + API Gateway WebSocket invokes once per message and cannot hold that stream, so
we run an always-on process. aiohttp gives us both the ALB health check (`/health`) and
the browser WebSocket (`/ws`) on one port.

BROWSER  ->  SERVER   messages (JSON):
    {"type": "start", "history": [...]}        # optional, begins a session
    {"type": "audio", "data": "<base64 pcm16 mono 16kHz>"}
    {"type": "audioEnd"}                        # user released the mic / finished a turn
    {"type": "stop"}                            # end the whole session

SERVER  ->  BROWSER   messages (JSON):
    {"type": "ready"}
    {"type": "transcript", "role": "USER"|"ASSISTANT", "text": "..."}
    {"type": "audio", "data": "<base64 pcm16 mono 24kHz>"}   # tutor speech to play
    {"type": "feedback", "category": "pronunciation"|"grammar", "text": "..."}  # side note
    {"type": "interrupted"}                     # barge-in: stop playing buffered audio
    {"type": "error", "message": "..."}
    {"type": "done"}

The "without interrupting" behavior comes from the `log_feedback` tool: the model keeps
talking naturally, and any correction is delivered silently via a tool call that we
forward to the browser as a `feedback` note instead of being spoken.
"""

import asyncio
import base64
import json
import os
import uuid

import boto3
from aiohttp import web, WSMsgType

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import (
    Config, HTTPAuthSchemeResolver, SigV4AuthScheme,
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver

MODEL_ID = os.environ.get("NOVA_SONIC_MODEL_ID", "amazon.nova-2-sonic-v1:0")
REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful Spanish tutor.")
VOICE_ID = os.environ.get("NOVA_SONIC_VOICE", "lupe")

# KB retrieval uses plain boto3 (this is a normal request/response call, unlike the
# bidirectional stream which needs the experimental SDK).
_bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=REGION)

# Tools the model can call. `search_materials` = RAG over course content.
# `log_feedback` = deliver a correction WITHOUT speaking it (keeps conversation flowing).
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "search_materials",
                "description": ("Search the learner's actual course materials: vocab lists, "
                                "lesson plans, assignments, syllabus. Use for any class question."),
                "inputSchema": {"json": json.dumps({
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "What to search for"}},
                    "required": ["query"],
                })},
            }
        },
        {
            "toolSpec": {
                "name": "log_feedback",
                "description": ("Silently record a pronunciation or grammar correction for the "
                                "learner to read. Does NOT speak it aloud. Use this instead of "
                                "correcting out loud, so the spoken conversation is not interrupted."),
                "inputSchema": {"json": json.dumps({
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["pronunciation", "grammar"]},
                        "note": {"type": "string", "description": "Short friendly English note"},
                    },
                    "required": ["category", "note"],
                })},
            }
        },
    ]
}


def retrieve_from_kb(query):
    if not KNOWLEDGE_BASE_ID or KNOWLEDGE_BASE_ID == "SET_VIA_SETUP_KB":
        return "Knowledge base not configured."
    try:
        resp = _bedrock_agent.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID, retrievalQuery={"text": query})
        results = resp.get("retrievalResults", [])
        return "\n\n".join(r["content"]["text"] for r in results) if results else "No materials found."
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


class NovaSonicSession:
    """One browser WebSocket <-> one Nova Sonic bidirectional stream."""

    def __init__(self, ws):
        self.ws = ws
        self.client = None
        self.stream = None
        self.active = False
        self.prompt_name = str(uuid.uuid4())
        self.sys_content = str(uuid.uuid4())
        self.audio_content = str(uuid.uuid4())
        self.role = None
        self.display_text = False
        # tool-call assembly
        self._tool_name = None
        self._tool_use_id = None
        self._tool_input = None

    # ---- low-level event I/O -------------------------------------------------
    def _init_client(self):
        # The experimental SDK's EnvironmentCredentialsResolver only reads AWS_* env
        # vars. On Fargate the task-role credentials arrive via the ECS container
        # credentials endpoint, which boto3 understands natively but this SDK does not.
        # So resolve them with boto3 and export them for the session. Task-role creds
        # remain valid well beyond Nova Sonic's ~8 min connection cap.
        creds = boto3.Session().get_credentials()
        if creds:
            frozen = creds.get_frozen_credentials()
            os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                os.environ["AWS_SESSION_TOKEN"] = frozen.token
        os.environ.setdefault("AWS_DEFAULT_REGION", REGION)

        # Param names follow the current Nova 2 Sonic getting-started guide. Older
        # aws_sdk_bedrock_runtime builds use http_auth_scheme_resolver / http_auth_schemes.
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{REGION}.amazonaws.com",
            region=REGION,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
        )
        self.client = BedrockRuntimeClient(config=config)

    async def _send(self, event: dict):
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps(event).encode("utf-8")))
        await self.stream.input_stream.send(chunk)

    async def _to_browser(self, obj: dict):
        if not self.ws.closed:
            await self.ws.send_str(json.dumps(obj))

    # ---- session lifecycle ---------------------------------------------------
    async def start(self):
        if not self.client:
            self._init_client()
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID))
        self.active = True

        await self._send({"event": {"sessionStart": {"inferenceConfiguration": {
            "maxTokens": 1024, "topP": 0.9, "temperature": 0.7}}}})

        await self._send({"event": {"promptStart": {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": 24000, "sampleSizeBits": 16,
                "channelCount": 1, "voiceId": VOICE_ID, "encoding": "base64", "audioType": "SPEECH"},
            "toolUseOutputConfiguration": {"mediaType": "application/json"},
            "toolConfiguration": TOOL_CONFIG,
        }}})

        # System prompt as a SYSTEM text turn.
        await self._send({"event": {"contentStart": {
            "promptName": self.prompt_name, "contentName": self.sys_content,
            "type": "TEXT", "interactive": True, "role": "SYSTEM",
            "textInputConfiguration": {"mediaType": "text/plain"}}}})
        await self._send({"event": {"textInput": {
            "promptName": self.prompt_name, "contentName": self.sys_content,
            "content": SYSTEM_PROMPT}}})
        await self._send({"event": {"contentEnd": {
            "promptName": self.prompt_name, "contentName": self.sys_content}}})

        # Open the continuous USER audio content block.
        await self._send({"event": {"contentStart": {
            "promptName": self.prompt_name, "contentName": self.audio_content,
            "type": "AUDIO", "interactive": True, "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": 16000, "sampleSizeBits": 16,
                "channelCount": 1, "audioType": "SPEECH", "encoding": "base64"}}}})

        asyncio.create_task(self._pump_responses())
        await self._to_browser({"type": "ready"})

    async def send_audio(self, audio_b64: str):
        if not self.active:
            return
        await self._send({"event": {"audioInput": {
            "promptName": self.prompt_name, "contentName": self.audio_content,
            "content": audio_b64}}})

    async def stop(self):
        if not self.active:
            return
        self.active = False
        try:
            await self._send({"event": {"contentEnd": {
                "promptName": self.prompt_name, "contentName": self.audio_content}}})
            await self._send({"event": {"promptEnd": {"promptName": self.prompt_name}}})
            await self._send({"event": {"sessionEnd": {}}})
            await self.stream.input_stream.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- tool result protocol ------------------------------------------------
    async def _send_tool_result(self, tool_use_id: str, result_text: str):
        content_name = str(uuid.uuid4())
        await self._send({"event": {"contentStart": {
            "promptName": self.prompt_name, "contentName": content_name,
            "interactive": False, "type": "TOOL", "role": "TOOL",
            "toolResultInputConfiguration": {
                "toolUseId": tool_use_id, "type": "TEXT",
                "textInputConfiguration": {"mediaType": "text/plain"}}}}})
        await self._send({"event": {"toolResult": {
            "promptName": self.prompt_name, "contentName": content_name,
            "content": result_text}}})
        await self._send({"event": {"contentEnd": {
            "promptName": self.prompt_name, "contentName": content_name}}})

    async def _handle_tool(self):
        name, tool_id, tool_input = self._tool_name, self._tool_use_id, (self._tool_input or {})
        if name == "search_materials":
            result = retrieve_from_kb(tool_input.get("query", ""))
            await self._send_tool_result(tool_id, result)
        elif name == "log_feedback":
            # Non-interrupting: forward as an on-screen note; the model keeps speaking.
            await self._to_browser({
                "type": "feedback",
                "category": tool_input.get("category", "grammar"),
                "text": tool_input.get("note", "")})
            await self._send_tool_result(tool_id, json.dumps({"acknowledged": True}))
        else:
            await self._send_tool_result(tool_id, json.dumps({"error": f"unknown tool {name}"}))
        self._tool_name = self._tool_use_id = self._tool_input = None

    # ---- read loop from Nova Sonic ------------------------------------------
    async def _pump_responses(self):
        try:
            while self.active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if not (result.value and result.value.bytes_):
                    continue
                data = json.loads(result.value.bytes_.decode("utf-8"))
                event = data.get("event", {})

                if "contentStart" in event:
                    cs = event["contentStart"]
                    self.role = cs.get("role")
                    self.display_text = False
                    extra = cs.get("additionalModelFields")
                    if extra:
                        try:
                            if json.loads(extra).get("generationStage") == "SPECULATIVE":
                                self.display_text = True
                        except json.JSONDecodeError:
                            pass

                elif "textOutput" in event:
                    text = event["textOutput"]["content"]
                    if '{ "interrupted" : true }' in text or '"interrupted":true' in text.replace(" ", ""):
                        await self._to_browser({"type": "interrupted"})
                        continue
                    await self._to_browser({"type": "transcript", "role": self.role or "ASSISTANT", "text": text})

                elif "audioOutput" in event:
                    await self._to_browser({"type": "audio", "data": event["audioOutput"]["content"]})

                elif "toolUse" in event:
                    tu = event["toolUse"]
                    self._tool_name = tu.get("toolName")
                    self._tool_use_id = tu.get("toolUseId")
                    raw = tu.get("content")
                    try:
                        self._tool_input = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    except json.JSONDecodeError:
                        self._tool_input = {}

                elif "contentEnd" in event and event.get("contentEnd", {}).get("type") == "TOOL":
                    await self._handle_tool()

                elif "completionEnd" in event:
                    await self._to_browser({"type": "done"})
        except Exception as e:  # noqa: BLE001
            await self._to_browser({"type": "error", "message": str(e)})
        finally:
            self.active = False


# ---- HTTP / WebSocket endpoints ---------------------------------------------
async def health(_request):
    return web.Response(text="ok")


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
    await ws.prepare(request)
    session = NovaSonicSession(ws)
    try:
        await session.start()
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            mtype = data.get("type")
            if mtype == "audio":
                await session.send_audio(data.get("data", ""))
            elif mtype in ("stop", "audioEnd"):
                if mtype == "stop":
                    break
    except Exception as e:  # noqa: BLE001
        await session._to_browser({"type": "error", "message": str(e)})
    finally:
        await session.stop()
        if not ws.closed:
            await ws.close()
    return ws


def make_app():
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/ws", ws_handler)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=8080)
