"""
Conversational Spanish bot client.

Audio flows:
  1. Mic → Transcribe Streaming (real-time STT, client-side)
  2. Recorded WAV → presigned S3 URL (upload)
  3. POST /converse (backend does SageMaker + Bedrock + KB + Polly)
  4. Response audio URL → download and play

Usage:
    pip install amazon-transcribe pyaudio boto3 requests
    python converse_client.py
"""

import asyncio
import base64
import io
import json
import os
import wave

import boto3
import pyaudio
import requests
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
REGION = "us-west-2"


class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.transcript = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.transcript += alt.transcript + " "


def get_stack_outputs():
    cf = boto3.client("cloudformation", region_name=REGION)
    resp = cf.describe_stacks(StackName="PronunciationCheckerStack")
    return {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}


def play_audio_url(url):
    import subprocess
    import tempfile
    data = requests.get(url).content
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data)
        f.flush()
        subprocess.run(["afplay", f.name])


async def record_and_transcribe():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n🎤 (press Enter when done)")
    frames = []
    stop_event = asyncio.Event()

    client = TranscribeStreamingClient(region=REGION)
    ts_stream = await client.start_stream_transcription(
        language_code="es-US", media_sample_rate_hz=RATE, media_encoding="pcm"
    )
    handler = TranscriptHandler(ts_stream.output_stream)

    async def wait_key():
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ch = await asyncio.get_event_loop().run_in_executor(None, lambda: os.read(fd, 1))
                if ch == b'\x1b':
                    print("\n👋 ¡Hasta luego!")
                    os._exit(0)
                if ch in (b'\n', b'\r'):
                    stop_event.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    async def feed_audio():
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            await ts_stream.input_stream.send_audio_event(audio_chunk=data)
            await asyncio.sleep(0)
        await ts_stream.input_stream.end_stream()

    enter_task = asyncio.create_task(wait_key())
    feed_task = asyncio.create_task(feed_audio())
    await stop_event.wait()
    await feed_task
    await handler.handle_events()
    enter_task.cancel()

    stream.stop_stream()
    stream.close()
    p.terminate()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return handler.transcript.strip(), buf.getvalue()


async def main():
    print("🇪🇸 ¡Hola! Let's chat in Spanish.")
    print("=" * 40)
    print("Speak naturally — I'll reply in Spanish.")
    print("Pronunciation notes appear as text below.")
    print("Say 'salir' or Ctrl+C to quit.\n")

    outputs = get_stack_outputs()
    api_url = outputs["ApiUrl"]

    history = []

    while True:
        text, audio_wav = await record_and_transcribe()
        if not text:
            print("  (no speech detected)")
            continue
        if text.lower().strip() in ("quit", "salir"):
            break

        print(f"  Tú: {text}")

        # 1. Get presigned upload URL
        resp = requests.get(f"{api_url}/upload-url")
        upload_info = resp.json()

        # 2. Upload audio to S3
        requests.put(upload_info["upload_url"], data=audio_wav,
                    headers={"Content-Type": "audio/wav"})

        # 3. Call /converse
        resp = requests.post(f"{api_url}/converse", json={
            "text": text,
            "audio_key": upload_info["key"],
            "history": history,
        })
        if resp.status_code != 200:
            print(f"  ❌ Server error ({resp.status_code}): {resp.text}")
            continue
        result = resp.json()

        if "error" in result:
            print(f"  ❌ {result['error']}")
            continue

        reply = result.get("reply", "")
        print(f"  Bot: {reply}")
        if result.get("pronunciation_note"):
            print(f"  📝 {result['pronunciation_note']}")
        if result.get("grammar_note"):
            print(f"  ✏️  {result['grammar_note']}")

        # 4. Play response audio
        if result.get("reply_audio_url"):
            play_audio_url(result["reply_audio_url"])

        # Maintain history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda *_: (print("\n👋 ¡Hasta luego!"), os._exit(0)))
    asyncio.run(main())
