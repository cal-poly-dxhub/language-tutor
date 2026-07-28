#!/usr/bin/env python3
"""
Nova Sonic probe — reproduces the bridge's exact event sequence outside the container,
printing every raw event the model returns.

Why this exists: when the bridge opens a stream successfully but the model answers
nothing, the interesting information is the raw event stream, and inside Fargate that is
only visible in CloudWatch. This runs the same sequence locally against Bedrock with
your own credentials and prints everything, including validation errors that the bridge
would otherwise only log.

    python3.12 -m venv .probe && .probe/bin/pip install -r bridge/requirements.txt
    .probe/bin/python tools/sonic_probe.py

It speaks a short sentence using macOS `say` (override with --wav), streams it at the
real-time cadence, then sends silence so the model endpoints the turn.

Options:
    --record SECS     record from your mic and send that (tests YOUR voice, not a
                      synthesised one — transcription can differ)
    --wav FILE        stream this file (repeatable: several turns in one session)
    --text "…"        cross-modal text input instead of audio
    --region REGION   default us-west-2
    --model ID        default amazon.nova-2-sonic-v1:0
    --endpointing X   HIGH | MEDIUM | LOW   (default LOW, matching the bridge)
    --no-tools        omit toolConfiguration from promptStart
    --silence SECS    trailing silence, default 4
"""

import argparse
import asyncio
import base64
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import uuid
import wave

import boto3
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

FRAME_SAMPLES = 512          # ~32ms at 16kHz, the cadence the docs ask for
SYSTEM_PROMPT = ("You are a warm Spanish conversation partner. Reply in short spoken "
                 "Spanish sentences. Never correct the learner out loud.")

T0 = time.monotonic()


def log(*parts):
    print(f"[{time.monotonic() - T0:6.2f}s]", *parts, flush=True)


def make_speech(text: str) -> bytes:
    """Render `text` to PCM16 mono 16kHz using macOS say + afconvert."""
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = os.path.join(tmp, "s.aiff"), os.path.join(tmp, "s.wav")
        voice = ["-v", "Paulina"]
        if subprocess.run(["say"] + voice + [text, "-o", aiff],
                          capture_output=True).returncode != 0:
            subprocess.run(["say", text, "-o", aiff], check=True, capture_output=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        aiff, wav], check=True, capture_output=True)
        return read_wav(wav)


def record(seconds: float) -> bytes:
    """
    Capture PCM16 mono 16kHz from the default microphone.

    This exists because synthetic speech is not representative: macOS `say` produces
    unusually clean, neutral audio, and the model's transcription behaviour can differ
    with a real voice. Testing with the actual speaker's audio is the only way to tell a
    model behaviour from a client bug.
    """
    try:
        import pyaudio
    except ImportError:
        sys.exit("--record needs pyaudio:  pip install pyaudio")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                     frames_per_buffer=FRAME_SAMPLES)
    print(f"\n>>> Recording {seconds:g}s — speak now (in whichever language you want to "
          f"test)…", flush=True)
    frames = []
    for _ in range(int(16000 * seconds / FRAME_SAMPLES)):
        frames.append(stream.read(FRAME_SAMPLES, exception_on_overflow=False))
    print(">>> done\n", flush=True)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    return b"".join(frames)


def read_wav(path: str) -> bytes:
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "need PCM16 mono"
        assert w.getframerate() == 16000, f"need 16kHz, got {w.getframerate()}"
        return w.readframes(w.getnframes())


def export_credentials(region: str) -> None:
    creds = boto3.Session().get_credentials()
    if creds is None:
        sys.exit("no AWS credentials found")
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    os.environ.setdefault("AWS_DEFAULT_REGION", region)


def build_config(region: str) -> Config:
    common = dict(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    resolver, schemes = HTTPAuthSchemeResolver(), {
        "aws.auth#sigv4": SigV4AuthScheme(service="bedrock")}
    try:
        return Config(auth_scheme_resolver=resolver, auth_schemes=schemes, **common)
    except TypeError:
        return Config(http_auth_scheme_resolver=resolver,
                      http_auth_schemes=schemes, **common)


TOOL_CONFIG = {"tools": [{"toolSpec": {
    "name": "search_materials",
    "description": "Search the learner's course materials.",
    "inputSchema": {"json": json.dumps({
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]})},
}}]}


class Probe:
    def __init__(self, args):
        self.args = args
        self.prompt = str(uuid.uuid4())
        self.sys_content = str(uuid.uuid4())
        self.audio_content = str(uuid.uuid4())
        self.stream = None
        self.events = {}
        self.pending_tool = None

    async def send(self, event: dict, label: str = ""):
        await self.stream.input_stream.send(
            InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(
                    bytes_=json.dumps(event).encode("utf-8"))))
        if label:
            log("-> ", label)

    async def run(self):
        a = self.args
        export_credentials(a.region)
        client = BedrockRuntimeClient(config=build_config(a.region))
        log(f"opening bidirectional stream: model={a.model} region={a.region}")
        self.stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=a.model))
        log("stream open")

        reader = asyncio.create_task(self.read_events())

        await self.send({"event": {"sessionStart": {
            "inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9,
                                       "temperature": 0.7},
            "turnDetectionConfiguration": {"endpointingSensitivity": a.endpointing},
        }}}, "sessionStart")

        prompt_start = {
            "promptName": self.prompt,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": 24000,
                "sampleSizeBits": 16, "channelCount": 1, "voiceId": a.voice,
                "encoding": "base64", "audioType": "SPEECH"},
        }
        if not a.no_tools:
            prompt_start["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt_start["toolConfiguration"] = TOOL_CONFIG
        await self.send({"event": {"promptStart": prompt_start}}, "promptStart")

        for label, event in [
            ("contentStart SYSTEM", {"contentStart": {
                "promptName": self.prompt, "contentName": self.sys_content,
                "type": "TEXT", "interactive": False, "role": "SYSTEM",
                "textInputConfiguration": {"mediaType": "text/plain"}}}),
            ("textInput SYSTEM", {"textInput": {
                "promptName": self.prompt, "contentName": self.sys_content,
                "content": SYSTEM_PROMPT}}),
            ("contentEnd SYSTEM", {"contentEnd": {
                "promptName": self.prompt, "contentName": self.sys_content}}),
        ]:
            await self.send({"event": event}, label)

        for turn in (a.history or []):
            role, _, text = turn.partition(":")
            content = str(uuid.uuid4())
            for label, event in [
                ("contentStart history " + role, {"contentStart": {
                    "promptName": self.prompt, "contentName": content, "type": "TEXT",
                    "interactive": False, "role": role.strip().upper(),
                    "textInputConfiguration": {"mediaType": "text/plain"}}}),
                ("textInput history", {"textInput": {
                    "promptName": self.prompt, "contentName": content,
                    "content": text.strip()}}),
                ("contentEnd history", {"contentEnd": {
                    "promptName": self.prompt, "contentName": content}}),
            ]:
                await self.send({"event": event}, label)

        if a.text:
            content = str(uuid.uuid4())
            for label, event in [
                ("contentStart USER text", {"contentStart": {
                    "promptName": self.prompt, "contentName": content, "type": "TEXT",
                    "interactive": True, "role": "USER",
                    "textInputConfiguration": {"mediaType": "text/plain"}}}),
                ("textInput USER", {"textInput": {
                    "promptName": self.prompt, "contentName": content,
                    "content": a.text}}),
                ("contentEnd USER text", {"contentEnd": {
                    "promptName": self.prompt, "contentName": content}}),
            ]:
                await self.send({"event": event}, label)
        else:
            utterances = []
            if a.record:
                utterances.append(("mic recording", record(a.record)))
            for path in (a.wav or []):
                utterances.append((path, read_wav(path)))
            if not utterances:
                utterances.append(("say: " + a.say, make_speech(a.say)))

            await self.send({"event": {"contentStart": {
                "promptName": self.prompt, "contentName": self.audio_content,
                "type": "AUDIO", "interactive": True, "role": "USER",
                "audioInputConfiguration": {
                    "mediaType": "audio/lpcm", "sampleRateHertz": 16000,
                    "sampleSizeBits": 16, "channelCount": 1, "audioType": "SPEECH",
                    "encoding": "base64"}}}}, "contentStart AUDIO")

            step = FRAME_SAMPLES * 2
            sil = base64.b64encode(b"\x00\x00" * FRAME_SAMPLES).decode()
            for index, (label, pcm) in enumerate(utterances, start=1):
                log(f"=== utterance {index}/{len(utterances)}: {label} "
                    f"({len(pcm)/32000:.2f}s) ===")
                for i in range(0, len(pcm), step):
                    await self.send({"event": {"audioInput": {
                        "promptName": self.prompt, "contentName": self.audio_content,
                        "content": base64.b64encode(pcm[i:i + step]).decode()}}})
                    await asyncio.sleep(0.032)
                # Trailing silence ends the turn; the gap then lets the tutor answer
                # before the next utterance, which is what makes this a real multi-turn
                # session rather than one long one.
                for _ in range(int(a.silence / 0.032)):
                    await self.send({"event": {"audioInput": {
                        "promptName": self.prompt, "contentName": self.audio_content,
                        "content": sil}}})
                    await asyncio.sleep(0.032)
                if index < len(utterances):
                    log(f"-> waiting {a.gap}s for the reply before the next utterance")
                    for _ in range(int(a.gap / 0.032)):
                        await self.send({"event": {"audioInput": {
                            "promptName": self.prompt,
                            "contentName": self.audio_content, "content": sil}}})
                        await asyncio.sleep(0.032)

            if a.close_audio:
                await self.send({"event": {"contentEnd": {
                    "promptName": self.prompt,
                    "contentName": self.audio_content}}}, "contentEnd AUDIO")

        log(f"waiting {a.wait}s for events…")
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=a.wait)
        except asyncio.TimeoutError:
            pass

        # Close the stream and keep reading briefly. This matters: a rejected
        # connection (bad credentials, missing model access, no egress) produces NO
        # events and NO error while the stream is open — the exception only surfaces
        # when the stream is torn down. Without this step the probe, and the bridge,
        # both just look "silent".
        log("closing the stream to force any latent error to surface…")
        try:
            if not a.text:
                await self.send({"event": {"contentEnd": {
                    "promptName": self.prompt, "contentName": self.audio_content}}})
            await self.send({"event": {"promptEnd": {"promptName": self.prompt}}})
            await self.send({"event": {"sessionEnd": {}}})
            await self.stream.input_stream.close()
        except Exception as exc:  # noqa: BLE001
            log("!! error while closing:", type(exc).__name__, exc)
        try:
            await asyncio.wait_for(reader, timeout=15)
        except asyncio.TimeoutError:
            log("reader still idle after close")
        except Exception as exc:  # noqa: BLE001
            log("!! reader raised on close:", type(exc).__name__, exc)

        log("event tally:", self.events or "NOTHING RECEIVED")
        if not self.events:
            log("=> the model returned no events at all for this sequence")
            log("   If an error appeared above, that is the real cause: a rejected")
            log("   connection stays silent until teardown.")

    async def send_tool_result(self, tool_use_id: str):
        content = str(uuid.uuid4())
        await self.send({"event": {"contentStart": {
            "promptName": self.prompt, "contentName": content, "interactive": False,
            "type": "TOOL", "role": "TOOL",
            "toolResultInputConfiguration": {
                "toolUseId": tool_use_id, "type": "TEXT",
                "textInputConfiguration": {"mediaType": "text/plain"}}}}})
        await self.send({"event": {"toolResult": {
            "promptName": self.prompt, "contentName": content,
            # Must be a stringified JSON object; prose fails the stream with
            # "ValidationException: Tool Response parsing error".
            "content": json.dumps({"result": self.args.tool_result})}}},
            "toolResult (stub)")
        await self.send({"event": {"contentEnd": {
            "promptName": self.prompt, "contentName": content}}})

    async def read_events(self):
        audio = 0
        while True:
            try:
                output = await self.stream.await_output()
                result = await output[1].receive()
            except Exception as exc:  # noqa: BLE001
                log("!! stream error:", type(exc).__name__, exc)
                return
            if not (result.value and result.value.bytes_):
                continue
            raw = result.value.bytes_.decode("utf-8")
            try:
                event = json.loads(raw).get("event", {})
            except json.JSONDecodeError:
                log("<- unparseable:", raw[:200])
                continue
            for key in event:
                self.events[key] = self.events.get(key, 0) + 1
            if "audioOutput" in event:
                audio += 1
                if audio == 1:
                    log("<- audioOutput (tutor speech) STARTED")
                continue
            if "textOutput" in event:
                out = event["textOutput"]
                if os.environ.get("PROBE_RAW_TEXTOUTPUT"):
                    log("<- RAW textOutput keys=" + ",".join(sorted(out))
                        + " role=" + repr(out.get("role"))
                        + " content=" + repr(out.get("content", "")[:60]))
                else:
                    log(f"<- textOutput role={out.get('role')}: {out.get('content','')[:160]}")
                continue
            if "contentStart" in event:
                cs = event["contentStart"]
                log(f"<- contentStart type={cs.get('type')} role={cs.get('role')} "
                    f"extra={cs.get('additionalModelFields')}")
                continue
            if "toolUse" in event:
                tu = event["toolUse"]
                self.pending_tool = tu.get("toolUseId")
                log(f"<- toolUse {tu.get('toolName')} {str(tu.get('content'))[:120]}")
                continue
            if "contentEnd" in event:
                ce = event["contentEnd"]
                log(f"<- contentEnd type={ce.get('type')} stop={ce.get('stopReason')}")
                # Answer tool calls with a stub so the turn can complete; without this
                # the model waits forever and the probe sees no reply at all.
                if ce.get("type") == "TOOL" and self.pending_tool:
                    await self.send_tool_result(self.pending_tool)
                    self.pending_tool = None
                continue
            log("<-", json.dumps(event)[:220])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--model", default="amazon.nova-2-sonic-v1:0")
    p.add_argument("--voice", default="lupe")
    p.add_argument("--endpointing", default="LOW", choices=["HIGH", "MEDIUM", "LOW"])
    p.add_argument("--wav", action="append",
                   help="stream this WAV; repeat the flag to send several "
                        "utterances in ONE session (multi-turn drift only "
                        "shows up from the second turn onward)")
    p.add_argument("--record", type=float, metavar="SECONDS",
                   help="record this many seconds from your microphone and send that, "
                        "instead of synthesised speech — use this to test your own voice")
    p.add_argument("--text")
    p.add_argument("--say", default="Hola, me llamo Zach. Ayer yo ir al mercado.")
    p.add_argument("--silence", type=float, default=4.0)
    p.add_argument("--gap", type=float, default=8.0,
                   help="seconds of silence between utterances, to let the "
                        "tutor finish replying")
    p.add_argument("--wait", type=float, default=30.0)
    p.add_argument("--system-file",
                   help="use this rendered system prompt instead of the built-in one, "
                        "so prompt changes can be tested before deploying")
    p.add_argument("--tool-result",
                   default="Week 1: greetings and introductions. Week 2: the preterite. "
                           "Midterm in week 6. Office hours Tuesdays at 2pm.",
                   help="stub result returned for any tool call")
    p.add_argument("--history", action="append", metavar="ROLE:TEXT",
                   help="inject a conversation-history turn before the audio, e.g. "
                        "--history 'USER:hola' --history 'ASSISTANT:¡Hola! ¿Qué tal?' "
                        "— replayed history is how the browser client resumes a session")
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--close-audio", action="store_true",
                   help="send contentEnd for the audio block after the silence")
    args = p.parse_args()
    if args.system_file:
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = pathlib.Path(args.system_file).read_text(encoding="utf-8")
    asyncio.run(Probe(args).run())


if __name__ == "__main__":
    main()
