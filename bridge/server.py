"""
Nova 2 Sonic bridge server (runs on Fargate).

Bridges a client WebSocket <-> the Bedrock Nova Sonic bidirectional stream, and tees
the learner's audio into an asynchronous pronunciation/grammar coach.

WHY A SERVER AND NOT LAMBDA
---------------------------
`InvokeModelWithBidirectionalStream` is a persistent HTTP/2 event stream (8 minute
cap). API Gateway WebSocket + Lambda invokes once per message and cannot hold that
stream open, so this runs as an always-on process. aiohttp serves the ALB health
check (`/health`) and the client WebSocket (`/ws`) on one port.

THE TWO LANES
-------------
Fast lane  (Nova Sonic): speech in -> speech out, ~1s, native turn detection and
           barge-in. Nothing in the slow lane can delay it.
Slow lane  (coach.py):   for each COMPLETED learner utterance we hand the transcript
           plus the raw PCM to a background task. Wav2Vec2 scores phonemes, Claude
           decides whether anything is worth saying, and any note is pushed to the
           client as an on-screen annotation seconds later. The tutor never speaks
           a correction, so the conversation is never interrupted.

WHY THE COACH IS NOT A SONIC TOOL CALL
--------------------------------------
Making the coach a Sonic tool would be worse on four counts: (1) Sonic blocks on the
tool result, so grading would add latency to the very turn it is grading; (2) the
model decides when to call it, so coverage becomes non-deterministic; (3) the result
re-enters the conversation context, which invites the tutor to say the correction out
loud; (4) a tool call carries text only — the phoneme scorer needs the raw audio,
which never enters the model's context. Teeing the audio here keeps the two lanes
fully decoupled.

COMPLETE UTTERANCES ONLY
------------------------
Nova Sonic emits the learner's ASR transcript as one or more TEXT content blocks with
`role: USER`. Each block closes with `stopReason` `PARTIAL_TURN` or `END_TURN`. We
accumulate text across `PARTIAL_TURN` blocks and only dispatch to the coach on
`END_TURN`, so the coach always sees the whole thing the learner said in one go —
long or short — never a fragment. `endpointingSensitivity` defaults to LOW so Sonic
waits longer before declaring the turn over.

CLIENT -> SERVER (JSON)
    {"type": "start", "language": "spanish",
     "history": [{"role": "USER"|"ASSISTANT", "text": "..."}]}
    {"type": "audio", "data": "<base64 pcm16 mono 16kHz>"}
    {"type": "text",  "content": "..."}     # optional typed input (cross-modal)
    {"type": "stop"}

SERVER -> CLIENT (JSON)
    {"type": "ready"}
    {"type": "transcript", "role": "USER"|"ASSISTANT", "text": "...",
     "stage": "FINAL"|"SPECULATIVE"}
    {"type": "audio", "data": "<base64 pcm16 mono 24kHz>"}
    {"type": "coaching", "turn": "..."}                       # slow lane started
    {"type": "feedback", "category": "pronunciation"|"grammar",
     "text": "...", "turn": "..."}                            # slow lane result
    {"type": "coach_done", "turn": "...", "notes": n, "diag": {...}}   # n may be 0
    {"type": "interrupted"}                                   # barge-in: flush audio
    {"type": "turn_end"}
    {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hmac
import json
import logging
import os
import uuid

import boto3
from aiohttp import WSMsgType, web

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver

from coach import Coach, SAMPLE_RATE

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bridge")

MODEL_ID = os.environ.get("NOVA_SONIC_MODEL_ID", "amazon.nova-2-sonic-v1:0")
REGION = os.environ.get("BEDROCK_REGION", "us-west-2")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a warm conversation partner helping someone practise a language they are "
    "learning. Never correct them out loud.")
COACH_PROMPT = os.environ.get("COACH_PROMPT", "")
# Voice and prompts are supplied by the stack from languages/<name>.json — the bridge
# is language-agnostic and has no default of its own beyond Nova Sonic's.
VOICE_ID = os.environ.get("NOVA_SONIC_VOICE", "matthew")
# LOW = wait longer before ending the learner's turn. Language learners pause to
# think mid-sentence; HIGH would chop one utterance into several fragments.
ENDPOINTING = os.environ.get("ENDPOINTING_SENSITIVITY", "LOW")
OUTPUT_SAMPLE_RATE = 24000
MAX_HISTORY_TURNS = 20
# Opening the bidirectional stream should take well under a second. If it hasn't
# happened by now something is wrong (credentials, throttling, region) and the client
# deserves to hear about it instead of waiting.
START_TIMEOUT = float(os.environ.get("START_TIMEOUT_SECONDS", "20"))
# How much audio may be sent with zero events back before we call it broken.
SILENCE_ALARM_SECONDS = float(os.environ.get("SILENCE_ALARM_SECONDS", "25"))
# How many utterances may be under review at once. The slow lane is fire-and-forget, so
# without a bound a stalled SageMaker call would let tasks accumulate for as long
# as the learner keeps talking. Two is ample: reviewing takes 1-3s and turns are slower
# than that. Beyond the bound the oldest waiting turn is dropped rather than queued
# indefinitely — a note nobody sees for a minute is worth less than nothing.
MAX_CONCURRENT_COACHING = int(os.environ.get("MAX_CONCURRENT_COACHING", "2"))

# Cap the per-turn audio tee. 30s at 16kHz/16-bit is ~960KB; longer turns keep only
# their most recent audio so a stuck session cannot grow without bound.
MAX_TURN_SECONDS = int(os.environ.get("MAX_TURN_SECONDS", "30"))
MAX_TURN_BYTES = MAX_TURN_SECONDS * SAMPLE_RATE * 2

# Optional shared-secret gate. The ALB is internet-facing, so without this anyone
# who learns the DNS name can spend Bedrock tokens. Set by the stack from Secrets
# Manager; when empty the server runs open and says so loudly at boot.
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")

# Every language the stack shipped, so a client can pick one per session without a
# redeploy: {name: {label, voice, system, coach}}. Packed gzip+base64 to stay well
# inside the task-definition size limit.
def _load_language_bundles() -> dict:
    raw = os.environ.get("LANGUAGE_BUNDLES", "")
    if not raw:
        return {}
    try:
        return json.loads(gzip.decompress(base64.b64decode(raw)).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read LANGUAGE_BUNDLES: %s", exc)
        return {}


LANGUAGE_BUNDLES = _load_language_bundles()
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "")


def resolve_language(requested: str = "") -> tuple:
    """
    Returns (name, system_prompt, coach_prompt, voice_id) for the requested language.

    Falls back to the deploy-time default, then to the single-language environment
    variables, so the bridge still works if the bundles are absent or the client asks
    for something that was never shipped.
    """
    for candidate in (requested, DEFAULT_LANGUAGE):
        bundle = LANGUAGE_BUNDLES.get((candidate or "").strip().lower())
        if bundle:
            return (candidate.strip().lower(), bundle["system"], bundle["coach"],
                    bundle["voice"])
    return "", SYSTEM_PROMPT, COACH_PROMPT, VOICE_ID

_bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=REGION)

# Nova Sonic's only tool is course-material RAG. Pronunciation/grammar coaching is
# deliberately NOT a tool — see the module docstring.
TOOL_CONFIG = {
    "tools": [{
        "toolSpec": {
            "name": "search_materials",
            "description": ("Search the learner's actual course materials: vocab lists, "
                            "lesson plans, assignments, syllabus. Use for any question "
                            "about their class."),
            "inputSchema": {"json": json.dumps({
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"],
            })},
        }
    }]
}


def kb_configured() -> bool:
    return bool(KNOWLEDGE_BASE_ID) and KNOWLEDGE_BASE_ID != "SET_VIA_SETUP_KB"


def retrieve_from_kb(query: str) -> str:
    if not kb_configured():
        logger.warning("search_materials called but no Knowledge Base is wired up; "
                       "run setup_kb.py <KB_ID> to enable course-material lookups")
        return ("No course materials are available in this session. Say so plainly and "
                "offer to help without them.")
    try:
        resp = _bedrock_agent.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID, retrievalQuery={"text": query})
        results = resp.get("retrievalResults", [])
        logger.info("kb retrieve %r -> %d passage(s)", query[:60], len(results))
        if not results:
            return "No matching course materials were found for that."
        return "\n\n".join(r["content"]["text"] for r in results)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kb retrieve failed: %s", exc)
        return f"Error: {exc}"


def _build_config() -> Config:
    """
    Nova 2 renamed the auth kwargs (`auth_scheme_resolver`/`auth_schemes`); older
    aws_sdk_bedrock_runtime builds use the `http_auth_*` spelling. Try the current
    names first and fall back so either SDK build works.
    """
    common = dict(
        endpoint_uri=f"https://bedrock-runtime.{REGION}.amazonaws.com",
        region=REGION,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    resolver = HTTPAuthSchemeResolver()
    schemes = {"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")}
    try:
        return Config(auth_scheme_resolver=resolver, auth_schemes=schemes, **common)
    except TypeError:
        return Config(http_auth_scheme_resolver=resolver, http_auth_schemes=schemes, **common)


def _export_credentials() -> None:
    """
    EnvironmentCredentialsResolver only reads AWS_* env vars. On Fargate the task
    role arrives via the ECS container-credentials endpoint, which boto3 understands
    but this SDK does not — so resolve once with boto3 and export. Task-role
    credentials outlive Nova Sonic's 8 minute connection cap.

    Raises if nothing can be resolved: the bidirectional stream would otherwise stall
    while the SDK retries an unsignable request, leaving the client waiting in silence.
    """
    creds = boto3.Session().get_credentials()
    if creds is None:
        raise RuntimeError(
            "no AWS credentials available — the bridge needs a task role (or AWS_* "
            "environment variables) to open a Nova Sonic stream")
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    os.environ.setdefault("AWS_DEFAULT_REGION", REGION)


class SonicSession:
    """One client WebSocket <-> one Nova Sonic bidirectional stream."""

    def __init__(self, ws: web.WebSocketResponse):
        self.ws = ws
        self.client = None
        self.stream = None
        self.active = False
        self.started = False

        self.prompt_name = str(uuid.uuid4())
        self.system_content = str(uuid.uuid4())
        self.audio_content = str(uuid.uuid4())
        self._pump_task = None
        self._watchdog_task = None
        # Nova Sonic reuses one completionId for an entire session and emits
        # completionStart only once, so neither can key a turn. The bridge counts:
        #   turn_index  - learner turns, incremented on userSpeechStart
        #   reply_index - tutor replies, incremented on userSpeechEnd
        # Replies are counted on userSpeechEnd rather than userSpeechStart on purpose.
        # When the learner interrupts, the previous reply's FINAL transcript can still be
        # arriving; keying it on the learner's new turn moved it into a fresh bubble and
        # the reply appeared twice.
        self.turn_index = 0
        self.reply_index = 0

        # --- output-event bookkeeping ---
        self.role = None
        self.stage = None
        self._user_content_ids = set()  # open content blocks carrying learner ASR
        self._speech_ended = False      # set by userSpeechEnd
        self._tool = {}

        # --- language (resolved in start(); may differ per session) ---
        self.language = ""
        self.system_prompt = SYSTEM_PROMPT
        self.voice_id = VOICE_ID

        # --- slow lane ---
        # Rebuilt in start() once the language is known, so the coach reviews against
        # the right language's rules.
        self.coach = Coach(system_prompt=COACH_PROMPT, region=REGION)
        self._turn_pcm = bytearray()   # tee of learner audio for the current utterance
        self._turn_text = []           # ASR fragments accumulated across PARTIAL_TURNs
        self._coach_tasks = set()
        self._coach_slots = asyncio.Semaphore(MAX_CONCURRENT_COACHING)

        # --- diagnostics ---
        # Counted so a silent session can be told apart from a broken one: "audio went
        # in and no events came back" is a very different fault from "no audio arrived".
        self.audio_events_sent = 0
        self.audio_bytes_sent = 0
        self.sonic_events = {}
        self.last_error = None

    # ---- low-level I/O ------------------------------------------------------
    async def _send(self, event: dict) -> None:
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps(event).encode("utf-8")))
        await self.stream.input_stream.send(chunk)

    async def _to_client(self, obj: dict) -> None:
        if self.ws.closed:
            logger.debug("dropping %s: client socket already closed", obj.get("type"))
            return
        try:
            await self.ws.send_str(json.dumps(obj))
        except Exception as exc:  # noqa: BLE001
            # Never let a failed push escape into a caller's error path, but do not
            # hide it either: a silently dropped error message is how a client ends
            # up waiting forever.
            logger.warning("failed to push %s to client: %r", obj.get("type"), exc)

    # ---- lifecycle ----------------------------------------------------------
    async def start(self, history=None, language: str = "") -> None:
        if self.started:
            return
        self.started = True
        requested = language
        self.language, self.system_prompt, coach_prompt, self.voice_id = \
            resolve_language(language)
        bundle = LANGUAGE_BUNDLES.get(self.language) or {}
        self.coach = Coach(system_prompt=coach_prompt, region=REGION,
                           espeak_language=bundle.get("espeakLanguage", ""))
        logger.info("session language=%s voice=%s",
                    self.language or "(env default)", self.voice_id)
        _export_credentials()
        self.client = BedrockRuntimeClient(config=_build_config())
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID))
        self.active = True

        await self._send({"event": {"sessionStart": {
            "inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7},
            "turnDetectionConfiguration": {"endpointingSensitivity": ENDPOINTING},
        }}})

        await self._send({"event": {"promptStart": {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                "sampleSizeBits": 16, "channelCount": 1, "voiceId": VOICE_ID,
                "encoding": "base64", "audioType": "SPEECH"},
            "toolUseOutputConfiguration": {"mediaType": "application/json"},
            "toolConfiguration": TOOL_CONFIG,
        }}})

        await self._text_turn(self.system_content, "SYSTEM", self.system_prompt)

        # Conversation history may be sent only once, after the system prompt and
        # before audio begins. This is how a client resumes past the 8 minute cap.
        for entry in (history or [])[-MAX_HISTORY_TURNS:]:
            role = entry.get("role")
            text = (entry.get("text") or "").strip()
            if role in ("USER", "ASSISTANT") and text:
                await self._text_turn(str(uuid.uuid4()), role, text)

        # One continuous USER audio block for the whole session.
        await self._send({"event": {"contentStart": {
            "promptName": self.prompt_name, "contentName": self.audio_content,
            "type": "AUDIO", "interactive": True, "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": SAMPLE_RATE,
                "sampleSizeBits": 16, "channelCount": 1, "audioType": "SPEECH",
                "encoding": "base64"}}}})

        # Keep a strong reference: a task whose only reference is the event loop's
        # weak set can be garbage-collected mid-await, which silently kills the pump
        # (and swallows the error it was trying to report to the client).
        self._pump_task = asyncio.create_task(self._pump_responses())
        self._watchdog_task = asyncio.create_task(self._watch_for_silence())
        # Report what was actually resolved. resolve_language falls back to the deployed
        # default when a client asks for a language this build does not carry, and a
        # silent fallback is indistinguishable from the UI ignoring the selector.
        await self._to_client({"type": "ready", "language": self.language,
                              "requested": (requested or "").strip().lower(),
                              "voice": self.voice_id})

    async def _watch_for_silence(self) -> None:
        """
        Report the case where audio is flowing to Nova Sonic and nothing comes back.

        This is not hypothetical: a rejected bidirectional stream (missing model access,
        a denied IAM action, no egress) produces no events AND no exception for as long
        as the stream stays open — the error only materialises at teardown, by which
        point the client has usually gone. Without this the failure is indistinguishable
        from a quiet learner.
        """
        try:
            while self.active:
                await asyncio.sleep(2)
                if self.sonic_events or not self.audio_events_sent:
                    continue
                # ~32ms per event, so this is roughly SILENCE_ALARM_SECONDS of audio.
                if self.audio_events_sent >= SILENCE_ALARM_SECONDS * 31:
                    logger.error(
                        "sent %d audio frames to %s in %s and received no events at "
                        "all; tearing the stream down to surface the real error",
                        self.audio_events_sent, MODEL_ID, REGION)
                    await self._to_client({
                        "type": "error",
                        "message": (f"no response from {MODEL_ID} after "
                                    f"{self.audio_events_sent} audio frames — asking "
                                    f"AWS why…")})
                    # A rejected stream reports nothing until teardown, so close the
                    # input side: the read pump then raises the actual AccessDenied /
                    # ValidationException and forwards it to the client. Without this
                    # the true cause is only ever visible in CloudWatch.
                    try:
                        await self.stream.input_stream.close()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("close during alarm failed: %s", exc)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("silence watchdog failed: %s", exc)

    async def _text_turn(self, content_name: str, role: str, text: str,
                         interactive: bool = False) -> None:
        await self._send({"event": {"contentStart": {
            "promptName": self.prompt_name, "contentName": content_name,
            "type": "TEXT", "interactive": interactive, "role": role,
            "textInputConfiguration": {"mediaType": "text/plain"}}}})
        await self._send({"event": {"textInput": {
            "promptName": self.prompt_name, "contentName": content_name,
            "content": text}}})
        await self._send({"event": {"contentEnd": {
            "promptName": self.prompt_name, "contentName": content_name}}})

    async def send_audio(self, audio_b64: str) -> None:
        if not self.active:
            return
        # Tee first: the coach needs exactly what the model heard.
        try:
            self._turn_pcm.extend(base64.b64decode(audio_b64))
        except (ValueError, TypeError):
            return
        if len(self._turn_pcm) > MAX_TURN_BYTES:
            del self._turn_pcm[:len(self._turn_pcm) - MAX_TURN_BYTES]
        await self._send({"event": {"audioInput": {
            "promptName": self.prompt_name, "contentName": self.audio_content,
            "content": audio_b64}}})
        self.audio_events_sent += 1
        self.audio_bytes_sent += len(audio_b64)
        if self.audio_events_sent in (1, 32, 300):
            logger.info("audio -> nova sonic: %d events, %d b64 bytes",
                        self.audio_events_sent, self.audio_bytes_sent)

    async def send_text(self, content: str) -> None:
        """Cross-modal input: the learner types instead of speaking."""
        if not self.active or not content.strip():
            return
        await self._text_turn(str(uuid.uuid4()), "USER", content.strip(),
                              interactive=True)

    async def stop(self) -> None:
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

    # ---- tools --------------------------------------------------------------
    async def _send_tool_result(self, tool_use_id: str, result) -> None:
        """
        Return a tool result to the model.

        `content` must be a STRINGIFIED JSON OBJECT, not prose. Sending raw text — for
        example a Knowledge Base passage straight from `retrieve` — fails the whole
        stream with `ValidationException: Tool Response parsing error`, which killed
        every course-materials lookup.
        """
        if not isinstance(result, str) or not result.strip().startswith("{"):
            result = json.dumps({"result": result})
        content_name = str(uuid.uuid4())
        await self._send({"event": {"contentStart": {
            "promptName": self.prompt_name, "contentName": content_name,
            "interactive": False, "type": "TOOL", "role": "TOOL",
            "toolResultInputConfiguration": {
                "toolUseId": tool_use_id, "type": "TEXT",
                "textInputConfiguration": {"mediaType": "text/plain"}}}}})
        await self._send({"event": {"toolResult": {
            "promptName": self.prompt_name, "contentName": content_name,
            "content": result}}})
        await self._send({"event": {"contentEnd": {
            "promptName": self.prompt_name, "contentName": content_name}}})

    async def _handle_tool(self) -> None:
        name = self._tool.get("name")
        tool_id = self._tool.get("id")
        args = self._tool.get("input") or {}
        self._tool = {}
        if not tool_id:
            return
        if name == "search_materials":
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, retrieve_from_kb, args.get("query", ""))
            await self._send_tool_result(tool_id, result)
        else:
            await self._send_tool_result(tool_id, json.dumps({"error": f"unknown tool {name}"}))

    # ---- slow lane ----------------------------------------------------------
    def _dispatch_coach(self, transcript: str) -> None:
        """
        Hand one COMPLETE learner utterance to the coach and return immediately.
        """
        # Everything the review needs is snapshotted here and passed by value: the
        # audio, the transcript, and the turn id it belongs to. Nothing the coach reads
        # can be mutated by a later utterance, which is what makes concurrent reviews
        # safe without a queue or a broker.
        pcm = bytes(self._turn_pcm)
        self._turn_pcm.clear()
        if not transcript.strip():
            return
        turn_id = f"u{self.turn_index}"
        task = asyncio.create_task(self._coach_turn(transcript.strip(), pcm, turn_id))
        self._coach_tasks.add(task)
        task.add_done_callback(self._coach_tasks.discard)

    async def _coach_turn(self, transcript: str, pcm: bytes, turn_id: str) -> None:
        if self._coach_slots.locked():
            logger.info("coaching at capacity (%d); skipping review of %r",
                        MAX_CONCURRENT_COACHING, transcript[:40])
            return
        async with self._coach_slots:
            await self._review(transcript, pcm, turn_id)

    async def _review(self, transcript: str, pcm: bytes, turn_id: str) -> None:
        try:
            shown = transcript
            if not self.coach.should_analyze(shown, pcm):
                return
            await self._to_client({"type": "coaching", "turn": shown})
            notes, diagnostics = await self.coach.analyze(shown, pcm)
            for category, text in notes:
                await self._to_client({"type": "feedback", "category": category,
                                       "text": text, "turn": shown})
            # Lets the client resolve its pending indicator: zero notes is the
            # normal, good outcome and should be shown as such, not left spinning.
            await self._to_client({"type": "coach_done", "turn": shown,
                                   "notes": len(notes),
                                   "diag": diagnostics})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("coach turn failed: %s", exc)
            await self._to_client({"type": "coach_done", "turn": transcript, "notes": 0})

    # ---- Nova Sonic read loop ----------------------------------------------
    async def _pump_responses(self) -> None:
        failure = None
        try:
            while self.active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if not (result.value and result.value.bytes_):
                    continue
                event = json.loads(result.value.bytes_.decode("utf-8")).get("event", {})
                await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            failure = exc
            self.last_error = str(exc)
            logger.warning("response pump ended: %s", exc)
            await self._to_client({"type": "error", "message": str(exc)})
        finally:
            self.active = False
            # The Nova Sonic stream is gone, so this socket can never carry another
            # word. Close it instead of leaving the client listening to a dead
            # session — its reconnect path can then start a fresh stream.
            if failure is not None and not self.ws.closed:
                try:
                    await self.ws.close(code=1011, message=b"stream ended")
                except Exception:  # noqa: BLE001
                    pass

    async def _handle_event(self, event: dict) -> None:
        # Every event type is counted and logged once. If this log stays empty while
        # audio_events_sent climbs, the model is receiving audio and answering nothing —
        # which points at the event sequence, not at the client or the network.
        for key in event:
            seen = self.sonic_events.get(key, 0)
            self.sonic_events[key] = seen + 1
            if seen == 0:
                logger.info("nova sonic event: %s", key)

        # Nova 2 Sonic brackets the learner's speech with userSpeechStart/userSpeechEnd.
        # These, not content stop reasons, are the authoritative turn boundaries: the
        # ASR content block for a *complete* utterance still closes with
        # stopReason PARTIAL_TURN (verified against the live model), so gating on
        # END_TURN would mean the coach never ran at all.
        if "userSpeechStart" in event:
            self._speech_ended = False
            # A new exchange begins. Everything the tutor says from here until the next
            # userSpeechStart is one reply, however many content blocks it arrives in.
            self.turn_index += 1
            return

        if "userSpeechEnd" in event:
            self._speech_ended = True
            self.reply_index += 1
            await self._maybe_dispatch_coach()
            return

        if "contentStart" in event:
            cs = event["contentStart"]
            self.role = cs.get("role")
            self.stage = _generation_stage(cs)
            # Learner ASR: role USER + TEXT. Remember the block so its contentEnd
            # tells us whether the utterance is finished.
            if self.role == "USER" and cs.get("type", "TEXT") == "TEXT":
                self._user_content_ids.add(cs.get("contentId"))
            return

        if "textOutput" in event:
            out = event["textOutput"]
            text = out.get("content", "") or ""
            role = out.get("role") or self.role or "ASSISTANT"
            # Nova Sonic v1 signalled barge-in inline; Nova 2 uses contentEnd
            # stopReason. Handle both so either model id behaves.
            if '"interrupted"' in text.replace(" ", "") and "true" in text:
                await self._to_client({"type": "interrupted"})
                return
            if role == "USER":
                self._turn_text.append(text)
            await self._to_client({
                "type": "transcript", "role": role, "text": text,
                "stage": self.stage or "FINAL",
                "turnId": (f"a{self.reply_index}" if role == "ASSISTANT"
                           else f"u{self.turn_index}")})
            return

        if "audioOutput" in event:
            await self._to_client({"type": "audio", "data": event["audioOutput"]["content"]})
            return

        if "toolUse" in event:
            tu = event["toolUse"]
            raw = tu.get("content")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                parsed = {}
            self._tool = {"name": tu.get("toolName"), "id": tu.get("toolUseId"),
                          "input": parsed}
            return

        if "contentEnd" in event:
            ce = event["contentEnd"]
            ctype = ce.get("type")
            stop = ce.get("stopReason")
            if ctype == "TOOL":
                await self._handle_tool()
                return
            if stop == "INTERRUPTED":
                await self._to_client({"type": "interrupted"})
            # End of the tutor's spoken turn. This is the reliable signal: observed
            # against the live model, a turn's final event is the ASSISTANT audio block
            # closing with END_TURN, and completionEnd may not arrive at all. The
            # client needs it to close the current speech bubble, or every reply piles
            # into one.
            if ctype == "AUDIO" and stop == "END_TURN":
                await self._to_client({"type": "turn_end",
                                       "turnId": f"a{self.reply_index}"})
            # A learner ASR block closed. Its stop reason is not a reliable end-of-turn
            # signal (a complete utterance still reports PARTIAL_TURN), so completion is
            # decided by userSpeechEnd instead — this just marks that the text for the
            # block has all arrived.
            if ctype == "TEXT" and ce.get("contentId") in self._user_content_ids:
                self._user_content_ids.discard(ce.get("contentId"))
                await self._maybe_dispatch_coach()
            return

        if "completionEnd" in event:
            # Safety net: the learner's turn is unambiguously over once the model has
            # finished answering, so nothing accumulated is ever left unreviewed.
            await self._maybe_dispatch_coach()
            await self._to_client({"type": "turn_end",
                                   "turnId": f"a{self.reply_index}"})

    async def _maybe_dispatch_coach(self) -> None:
        """
        Hand the utterance to the coach once BOTH halves have arrived: the model has
        told us the learner stopped speaking, and the transcript text is in hand. The
        two can arrive in either order, so this is called from both places and only
        fires when it has everything.
        """
        if not (self._speech_ended and self._turn_text):
            return
        transcript = " ".join(t.strip() for t in self._turn_text if t.strip())
        self._turn_text = []
        self._speech_ended = False
        self._dispatch_coach(transcript)


def _generation_stage(content_start: dict):
    extra = content_start.get("additionalModelFields")
    if not extra:
        return None
    try:
        return json.loads(extra).get("generationStage")
    except (json.JSONDecodeError, TypeError):
        return None


# ---- HTTP / WebSocket -------------------------------------------------------

async def health(_request):
    return web.Response(text="ok")


def _authorized(request) -> bool:
    if not ACCESS_TOKEN:
        return True
    supplied = request.query.get("token") or ""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        supplied = supplied or header[len("Bearer "):]
    # Constant-time compare: this is a bearer secret.
    return hmac.compare_digest(supplied, ACCESS_TOKEN)


async def ws_handler(request):
    if not _authorized(request):
        return web.Response(status=401, text="unauthorized")

    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
    await ws.prepare(request)
    session = SonicSession(ws)

    async def ensure_started(history=None, language: str = "") -> bool:
        """
        Open the Nova Sonic stream, bounded in time. A stalled or failed start must
        surface as an error the client can show and retry on — never as silence.
        """
        if session.started:
            return True
        try:
            await asyncio.wait_for(session.start(history, language),
                                   timeout=START_TIMEOUT)
            return True
        except asyncio.TimeoutError:
            logger.error("nova sonic stream did not open within %ss", START_TIMEOUT)
            await session._to_client({
                "type": "error",
                "message": f"could not open the Nova Sonic stream within "
                           f"{START_TIMEOUT}s"})
        except Exception as exc:  # noqa: BLE001
            logger.error("nova sonic stream failed to open: %s", exc)
            await session._to_client({"type": "error", "message": str(exc)})
        return False

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            mtype = data.get("type")

            if mtype == "start":
                if not await ensure_started(data.get("history"),
                                            data.get("language", "")):
                    break
            elif mtype == "audio":
                if not await ensure_started():
                    break
                await session.send_audio(data.get("data", ""))
            elif mtype == "text":
                if not await ensure_started():
                    break
                await session.send_text(data.get("content", ""))
            elif mtype == "stop":
                break
            elif mtype == "ping":
                # Diagnostic round trip: reports what the bridge has actually sent to
                # and received from Nova Sonic. Carries no secrets.
                await session._to_client({
                    "type": "debug",
                    "started": session.started,
                    "active": session.active,
                    "audioEventsSent": session.audio_events_sent,
                    "audioBytesSent": session.audio_bytes_sent,
                    "sonicEvents": session.sonic_events,
                    "lastError": session.last_error,
                    "model": MODEL_ID,
                    "region": REGION,
                    "endpointing": ENDPOINTING,
                    "knowledgeBase": KNOWLEDGE_BASE_ID if kb_configured() else None,
                    "voice": session.voice_id,
                    "language": session.language,
                    "languagesAvailable": sorted(LANGUAGE_BUNDLES),
                })
    except Exception as exc:  # noqa: BLE001
        logger.warning("ws session error: %s", exc)
        await session._to_client({"type": "error", "message": str(exc)})
    finally:
        await session.stop()
        if not ws.closed:
            await ws.close()
    return ws


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/ws", ws_handler)
    return app


if __name__ == "__main__":
    if not kb_configured():
        logger.warning("no Knowledge Base configured (KNOWLEDGE_BASE_ID=%r) — "
                       "search_materials will find nothing until you run "
                       "setup_kb.py <KB_ID>", KNOWLEDGE_BASE_ID)
    if not ACCESS_TOKEN:
        logger.warning("ACCESS_TOKEN is not set — /ws is UNAUTHENTICATED and anyone "
                       "who can reach this load balancer can spend Bedrock tokens.")
    web.run_app(make_app(), host="0.0.0.0", port=8080)
