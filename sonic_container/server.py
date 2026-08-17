"""
Raw Nova Sonic bridge server (runs on Fargate).

Bridges a browser WebSocket <-> the Bedrock Nova Sonic bidirectional stream. This is the
RAW speech-to-speech mode of the tutor: STT + reasoning + TTS + turn-taking + barge-in
all happen inside one model. There is NO pronunciation/grammar coaching and NO course-
material tool here — that is entirely the job of the separate "custom bot" mode
(Transcribe + Wav2Vec2 + Bedrock + Polly). Keeping this bridge tool-free is deliberate:
"raw Sonic" means just the model, nothing bolted on.

WHY A SERVER (not Lambda)
-------------------------
Nova Sonic's `InvokeModelWithBidirectionalStream` is a persistent HTTP/2 event stream.
Lambda + API Gateway WebSocket invokes once per message and cannot hold that stream, so
we run an always-on process. aiohttp serves both the ALB health check (`/health`) and
the browser WebSocket (`/ws`) on one port.

ACCESS GATE
-----------
The ALB is internet-facing, so an open bridge would let anyone run conversations on your
Bedrock bill. Every WebSocket upgrade must carry the shared secret, either as `?token=`
or an `Authorization: Bearer <token>` header. The secret is injected as ACCESS_TOKEN.

BROWSER  ->  SERVER   messages (JSON):
    {"type": "start", "history": [{"role": "USER"|"ASSISTANT", "text": "..."}]}
    {"type": "audio", "data": "<base64 pcm16 mono 16kHz>"}
    {"type": "stop"}                            # end the whole session

SERVER  ->  BROWSER   messages (JSON):
    {"type": "ready"}
    {"type": "transcript", "role": "USER"|"ASSISTANT", "text": "...", "stage": "..."}
    {"type": "audio", "data": "<base64 pcm16 mono 24kHz>"}   # tutor speech to play
    {"type": "interrupted"}                     # barge-in: stop playing buffered audio
    {"type": "error", "message": "..."}
    {"type": "done"}
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
REGION = os.environ.get("BEDROCK_REGION", "us-west-2")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful Spanish tutor.")
VOICE_ID = os.environ.get("NOVA_SONIC_VOICE", "lupe")
# Shared secret required on the WebSocket upgrade. Empty means "no gate" (local dev only).
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
# How many prior turns to seed when a browser reconnects (Nova Sonic caps a connection at
# ~8 minutes, so the client replays recent history to continue seamlessly).
MAX_HISTORY = 20


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
        self.stage = None

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

        # aws_sdk_bedrock_runtime 0.1.x renamed these Config kwargs from the 0.0.x names
        # still shown in the AWS docs example: http_auth_scheme_resolver -> auth_scheme_
        # resolver and http_auth_schemes -> auth_schemes. requirements.txt pins the
        # matching 0.1.x smithy stack so this signature stays valid.
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{REGION}.amazonaws.com",
            region=REGION,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
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
    async def start(self, history=None):
        if not self.client:
            self._init_client()
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID))
        self.active = True

        await self._send({"event": {"sessionStart": {"inferenceConfiguration": {
            "maxTokens": 1024, "topP": 0.9, "temperature": 0.7}}}})

        # No toolConfiguration: raw Sonic has no tools.
        await self._send({"event": {"promptStart": {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": 24000, "sampleSizeBits": 16,
                "channelCount": 1, "voiceId": VOICE_ID, "encoding": "base64", "audioType": "SPEECH"},
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

        # Replay recent history so a reconnect (or the 8-minute cap) continues the
        # conversation instead of starting over. Each item is a completed text turn.
        for turn in (history or [])[-MAX_HISTORY:]:
            role = turn.get("role", "USER")
            text = (turn.get("text") or "").strip()
            if not text or role not in ("USER", "ASSISTANT"):
                continue
            cname = str(uuid.uuid4())
            await self._send({"event": {"contentStart": {
                "promptName": self.prompt_name, "contentName": cname,
                "type": "TEXT", "interactive": False, "role": role,
                "textInputConfiguration": {"mediaType": "text/plain"}}}})
            await self._send({"event": {"textInput": {
                "promptName": self.prompt_name, "contentName": cname, "content": text}}})
            await self._send({"event": {"contentEnd": {
                "promptName": self.prompt_name, "contentName": cname}}})

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
                    self.stage = None
                    extra = cs.get("additionalModelFields")
                    if extra:
                        try:
                            self.stage = json.loads(extra).get("generationStage")
                        except json.JSONDecodeError:
                            pass

                elif "textOutput" in event:
                    text = event["textOutput"]["content"]
                    if '{ "interrupted" : true }' in text or '"interrupted":true' in text.replace(" ", ""):
                        await self._to_browser({"type": "interrupted"})
                        continue
                    await self._to_browser({
                        "type": "transcript",
                        "role": self.role or "ASSISTANT",
                        "stage": self.stage,
                        "text": text})

                elif "audioOutput" in event:
                    await self._to_browser({"type": "audio", "data": event["audioOutput"]["content"]})

                elif "completionEnd" in event:
                    await self._to_browser({"type": "done"})
        except Exception as e:  # noqa: BLE001
            await self._to_browser({"type": "error", "message": str(e)})
        finally:
            self.active = False


# ---- HTTP / WebSocket endpoints ---------------------------------------------
async def health(_request):
    return web.Response(text="ok")


def _authorized(request) -> bool:
    if not ACCESS_TOKEN:
        return True  # no gate configured (local dev)
    token = request.query.get("token", "")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):]
    return token == ACCESS_TOKEN


async def ws_handler(request):
    if not _authorized(request):
        return web.Response(status=401, text="unauthorized")
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
    await ws.prepare(request)
    session = NovaSonicSession(ws)
    started = False
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            mtype = data.get("type")
            if mtype == "start":
                if not started:
                    started = True
                    await session.start(history=data.get("history"))
            elif mtype == "audio":
                if not started:
                    started = True
                    await session.start()
                await session.send_audio(data.get("data", ""))
            elif mtype == "stop":
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
