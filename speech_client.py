#!/usr/bin/env python3
"""
CLI client for the hybrid Language Tutor — speech in, speech out.

Just talk. There is no Enter key and no push-to-talk: Nova Sonic decides when your
turn ended and you can talk over it to interrupt. Coach notes (pronunciation/grammar)
print a few seconds after each sentence, without pausing the conversation.

    pip install pyaudio websockets boto3
    python3 speech_client.py

By default it reads the bridge URL and access token straight from the deployed stack
(CloudFormation outputs + Secrets Manager). Override with:

    TUTOR_WS_URL=ws://host/ws TUTOR_TOKEN=... python3 speech_client.py

Ctrl+C to quit.
"""

import asyncio
import base64
import json
import os
import queue
import signal
import sys
import threading

import boto3
import pyaudio
import websockets

STACK_NAME = os.environ.get("TUTOR_STACK", "LanguageTutorStack")
REGION = os.environ.get("TUTOR_REGION", os.environ.get("AWS_REGION", "us-west-2"))

IN_RATE = 16000       # what we send
OUT_RATE = 24000      # what Nova Sonic returns
FRAME = 512           # ~32ms per audioInput event
HISTORY_TURNS = 20


def resolve_endpoint():
    """Prefer env vars; otherwise read the stack outputs and the shared secret."""
    url = os.environ.get("TUTOR_WS_URL")
    token = os.environ.get("TUTOR_TOKEN")
    if url:
        return url, token

    cf = boto3.client("cloudformation", region_name=REGION)
    outputs = {o["OutputKey"]: o["OutputValue"]
               for o in cf.describe_stacks(StackName=STACK_NAME)["Stacks"][0]["Outputs"]}
    url = outputs["WsUrl"]
    if token is None:
        secret_name = outputs.get("AccessTokenSecretName")
        if secret_name:
            sm = boto3.client("secretsmanager", region_name=REGION)
            token = sm.get_secret_value(SecretId=secret_name)["SecretString"]
    return url, token


class Speaker:
    """
    Plays 24kHz PCM on a worker thread. `flush` exists for barge-in: Nova Sonic runs
    faster than real time, so on an interruption there is buffered audio here that
    must be dropped or the tutor talks over you.
    """

    def __init__(self):
        self.q = queue.Queue()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(format=pyaudio.paInt16, channels=1,
                                     rate=OUT_RATE, output=True)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                chunk = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._stream.write(chunk)
            except Exception:
                break

    def play(self, chunk: bytes):
        self.q.put(chunk)

    def flush(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def close(self):
        self._stop.set()
        self._thread.join(timeout=0.5)
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


class Mic:
    """Continuous capture pushed onto the event loop; never stops mid-conversation."""

    def __init__(self, loop):
        self.loop = loop
        self.q = asyncio.Queue()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(format=pyaudio.paInt16, channels=1, rate=IN_RATE,
                                     input=True, frames_per_buffer=FRAME)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data = self._stream.read(FRAME, exception_on_overflow=False)
            except Exception:
                break
            self.loop.call_soon_threadsafe(self.q.put_nowait, data)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=0.5)
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


CHIP = {"pronunciation": "🗣  pronunciation", "grammar": "✏️  grammar"}


class Session:
    def __init__(self):
        self.history = []
        self.bot_spec = ""
        self.bot_final = ""
        self.printed_bot = False

    def on_user(self, text):
        if text.strip():
            print(f"\n  Tú:  {text.strip()}")
            self.history.append({"role": "USER", "text": text.strip()})

    def on_bot(self, text, stage):
        if not text:
            return
        if stage == "FINAL":
            self.bot_final += text
        else:
            self.bot_spec += text
        if not self.printed_bot:
            print("  Bot: ", end="", flush=True)
            self.printed_bot = True
        print(text, end="", flush=True)

    def close_turn(self):
        said = (self.bot_final or self.bot_spec).strip()
        if said:
            self.history.append({"role": "ASSISTANT", "text": said})
        if self.printed_bot:
            print()
        self.bot_spec = self.bot_final = ""
        self.printed_bot = False


async def pump_mic(ws, mic):
    while True:
        chunk = await mic.q.get()
        await ws.send(json.dumps({"type": "audio",
                                  "data": base64.b64encode(chunk).decode()}))


async def pump_ws(ws, speaker, session):
    async for raw in ws:
        try:
            m = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = m.get("type")
        if t == "ready":
            print("🎤 Listening — just start talking (Ctrl+C to quit)")
        elif t == "transcript":
            if m.get("role") == "USER":
                session.on_user(m.get("text", ""))
            else:
                session.on_bot(m.get("text", ""), m.get("stage"))
        elif t == "audio":
            speaker.play(base64.b64decode(m["data"]))
        elif t == "interrupted":
            speaker.flush()
            session.close_turn()
            print("  ⏹  (interrupted)")
        elif t == "turn_end":
            session.close_turn()
        elif t == "coaching":
            print(f"  … reviewing “{m.get('turn','')}”")
        elif t == "feedback":
            label = CHIP.get(m.get("category"), m.get("category", "note"))
            print(f"  {label}: {m.get('text','')}")
        elif t == "coach_done" and not m.get("notes"):
            print("  ✓ nothing to fix")
        elif t == "error":
            print(f"  ❌ {m.get('message')}")


async def run_once(url, token, mic, speaker, session):
    """One Nova Sonic connection. Returns True if it should be resumed."""
    full = url + (f"?token={token}" if token else "")
    try:
        async with websockets.connect(full, max_size=None, ping_interval=20) as ws:
            await ws.send(json.dumps({"type": "start",
                                      "history": session.history[-HISTORY_TURNS:]}))
            mic_task = asyncio.create_task(pump_mic(ws, mic))
            try:
                await pump_ws(ws, speaker, session)
            finally:
                mic_task.cancel()
                await asyncio.gather(mic_task, return_exceptions=True)
        return True
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        print(f"  connection ended: {exc}")
        return True


async def main():
    url, token = resolve_endpoint()
    if not token:
        print("⚠️  No access token resolved — the bridge will reject the connection "
              "unless it was deployed without one.")
    print("🇪🇸 ¡Hola! Speech-to-speech Spanish practice.")
    print("=" * 52)

    loop = asyncio.get_event_loop()
    mic = Mic(loop)
    speaker = Speaker()
    session = Session()
    try:
        # Nova Sonic caps one connection at 8 minutes; reconnecting with the
        # transcript keeps the conversation going past it.
        for attempt in range(100):
            resume = await run_once(url, token, mic, speaker, session)
            if not resume:
                break
            speaker.flush()
            await asyncio.sleep(0.3)
    finally:
        mic.close()
        speaker.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: (print("\n👋 ¡Hasta luego!"), os._exit(0)))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
