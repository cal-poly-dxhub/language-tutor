"""
Conversational Spanish bot with pronunciation feedback.

You have a natural conversation in Spanish. The bot replies in Spanish (voice + text).
Pronunciation notes appear as a text side-channel — the bot doesn't interrupt the flow.

Usage:
    pip install amazon-transcribe pyaudio boto3 pydub simpleaudio
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
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
REGION = "us-west-2"

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system.txt")
with open(PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read()


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


def play_audio_b64(audio_b64):
    """Play base64 mp3 via macOS afplay (no extra deps)."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(base64.b64decode(audio_b64))
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

    return handler.transcript.strip(), base64.b64encode(buf.getvalue()).decode()


async def main():
    print("🇪🇸 ¡Hola! Let's chat in Spanish.")
    print("=" * 40)
    print("Speak naturally — I'll reply in Spanish.")
    print("Pronunciation notes appear as text below.")
    print("Say 'salir' or Ctrl+C to quit.\n")

    outputs = get_stack_outputs()
    endpoint_name = outputs["EndpointName"]

    sagemaker = boto3.client("sagemaker-runtime", region_name=REGION)
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    polly_client = boto3.client("polly", region_name=REGION)

    history = []

    while True:
        text, audio_b64 = await record_and_transcribe()
        if not text:
            print("  (no speech detected)")
            continue
        if text.lower().strip() in ("quit", "salir"):
            break

        print(f"  Tú: {text}")

        # 1. Phonemes from SageMaker
        sm_resp = sagemaker.invoke_endpoint(
            EndpointName=endpoint_name, ContentType="application/json",
            Body=json.dumps({"audio_b64": audio_b64, "text": text}),
        )
        phonemes = json.loads(sm_resp["Body"].read())
        actual = phonemes.get("actual_phonemes", [])
        expected = phonemes.get("expected_phonemes", [])

        # 2. Conversational reply from Bedrock
        user_msg = f'[The learner said: "{text}"]\n[Expected phonemes: {expected}]\n[Actual phonemes: {actual}]'
        history.append({"role": "user", "content": user_msg})

        br_resp = bedrock.invoke_model(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            contentType="application/json", accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "system": SYSTEM_PROMPT,
                "messages": history,
            }),
        )
        llm_text = json.loads(br_resp["body"].read())["content"][0]["text"]
        # Parse XML tags from response
        import re
        reply = re.search(r'<reply>(.*?)</reply>', llm_text, re.DOTALL)
        note = re.search(r'<pronunciation_note>(.*?)</pronunciation_note>', llm_text, re.DOTALL)
        grammar = re.search(r'<grammar_note>(.*?)</grammar_note>', llm_text, re.DOTALL)
        llm_output = {
            "reply": reply.group(1).strip() if reply else llm_text,
            "pronunciation_note": note.group(1).strip() if note and note.group(1).strip() else None,
            "grammar_note": grammar.group(1).strip() if grammar and grammar.group(1).strip() else None,
        }

        reply = llm_output["reply"]
        history.append({"role": "assistant", "content": json.dumps(llm_output)})

        # Keep history bounded
        if len(history) > 20:
            history = history[-20:]

        # 3. Speak reply in Spanish
        polly_resp = polly_client.synthesize_speech(
            Text=reply, OutputFormat="mp3", VoiceId="Lupe", Engine="neural", LanguageCode="es-US"
        )
        reply_audio = base64.b64encode(polly_resp["AudioStream"].read()).decode()

        print(f"  Bot: {reply}")
        if llm_output.get("pronunciation_note"):
            print(f"  📝 {llm_output['pronunciation_note']}")
        if llm_output.get("grammar_note"):
            print(f"  ✏️  {llm_output['grammar_note']}")

        play_audio_b64(reply_audio)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda *_: (print("\n👋 ¡Hasta luego!"), os._exit(0)))
    asyncio.run(main())
