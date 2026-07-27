"""
Streaming Spanish tutor client — no press-Enter.

Differences from converse_client.py (the HTTP path):
  - End-of-turn is detected automatically with voice-activity detection (VAD).
    You just stop talking; a short silence ends your turn.
  - The reply streams back over a WebSocket: text appears token-by-token and
    audio plays sentence-by-sentence as it is synthesized, instead of waiting for
    one buffered response.

This CLI mirrors the protocol a browser web client will use later (see README):
  same WebSocket messages, same VAD-based endpointing, same S3 upload for the
  full utterance used by pronunciation scoring.

Setup:
    pip install amazon-transcribe pyaudio boto3 requests websockets webrtcvad
    python converse_stream_client.py
"""

import asyncio
import base64
import io
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import wave
import array
import time
from collections import deque

import boto3
import pyaudio
import requests
import websockets
import webrtcvad
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
REGION = "us-west-2"

# VAD / endpointing tuning.
VAD_FRAME_MS = 30                              # webrtcvad accepts 10/20/30ms frames
FRAME_SAMPLES = RATE * VAD_FRAME_MS // 1000    # 480 samples @ 30ms
FRAME_BYTES = FRAME_SAMPLES * 2                # 960 bytes (16-bit mono)
VAD_AGGRESSIVENESS = 2                         # 0=lenient … 3=aggressive noise filter
PADDING_MS = 300                               # sustained-speech window to START a turn
TRIGGER_RATIO = 0.9                            # fraction of that window that must be voiced
SILENCE_MS = 800                               # trailing silence that ENDS a turn
MAX_UTTERANCE_MS = 30000                       # hard cap once a turn has started
RMS_FLOOR = 350                                # min loudness (0–32767) to count a frame as speech
# Barge-in (interrupting the bot). There's no echo cancellation, so the mic also
# hears the bot's own audio. Instead of a fixed loudness threshold (impossible to
# tune across volumes), we track the running bleed/ambient level and trigger when
# the user's voice is clearly above it.
BARGE_RMS_MIN = 400                            # absolute floor so silence never triggers
BARGE_FACTOR = 1.8                             # must be this many× the running bleed level
BARGE_PADDING_MS = 240                         # sustained speech required to interrupt
BARGE_TRIGGER_RATIO = 0.8
PREROLL_MS = 600                               # onset audio carried from a barge into the next turn
DEBUG_VAD = os.environ.get("KIRO_VAD_DEBUG") == "1"


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


PLAY_CMD = ["afplay"]  # overridable for testing


class AudioPlayer:
    """
    Plays streamed MP3 chunks sequentially in arrival order on a worker thread.
    Interruptible: stop() kills the current clip and drops anything queued so the
    bot goes quiet immediately when the user barges in.
    """

    def __init__(self):
        self.q = queue.Queue()
        self._proc = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            data = self.q.get()
            if data is None or self._stopped.is_set():
                continue
            path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(data)
                    f.flush()
                    path = f.name
                with self._lock:
                    if self._stopped.is_set():
                        continue
                    self._proc = subprocess.Popen(
                        PLAY_CMD + [path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._proc.wait()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._proc = None
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def reset(self):
        """Begin a fresh reply: allow playback again."""
        self._stopped.clear()

    def enqueue(self, data):
        self.q.put(data)

    def stop(self):
        """Interrupt immediately: kill current clip and drop the queue."""
        self._stopped.set()
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    def is_busy(self):
        with self._lock:
            playing = self._proc is not None and self._proc.poll() is None
        return playing or not self.q.empty()


class MicReader:
    """
    One microphone stream, read continuously on a background thread and pushed into
    an asyncio.Queue. Because the reader never stops, frames are buffered in the
    queue during Transcribe-stream startup or phase switches — so no audio is lost
    at the boundary between the bot speaking and the user's next turn (the bug that
    made interrupted words vanish). Consumers (capture / barge detector) take turns
    draining the queue; only one is active at a time.
    """

    def __init__(self, loop):
        self.loop = loop
        self.q = asyncio.Queue()
        self._stop = threading.Event()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                  input=True, frames_per_buffer=FRAME_SAMPLES)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data = self.stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            except Exception:
                break
            self.loop.call_soon_threadsafe(self.q.put_nowait, data)

    def drain(self):
        """Discard any buffered frames (e.g., stale playback tail)."""
        try:
            while True:
                self.q.get_nowait()
        except asyncio.QueueEmpty:
            pass

    def close(self):
        self._stop.set()
        self.thread.join(timeout=0.5)
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.p.terminate()


def _frame_rms(data):
    """RMS loudness of a 16-bit mono PCM frame (0–32767)."""
    n = len(data) // 2
    if n == 0:
        return 0.0
    a = array.array("h")
    a.frombytes(data[: n * 2])
    return (sum(s * s for s in a) / n) ** 0.5


async def capture(reader, preroll=None):
    """
    Pull frames from the mic queue until VAD detects end-of-turn; return
    (transcript, wav_bytes). `preroll` is the interrupt onset consumed by the barge
    detector — it's prepended so the interrupting words reach Transcribe, and
    capture starts already in the 'speaking' state. Frames that arrived after the
    barge are still waiting in the queue, so nothing between them is lost.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frames = []
    stats = {"n": 0, "voiced": 0, "maxrms": 0.0}

    client = TranscribeStreamingClient(region=REGION)
    ts_stream = await client.start_stream_transcription(
        language_code="es-US", media_sample_rate_hz=RATE, media_encoding="pcm")
    handler = TranscriptHandler(ts_stream.output_stream)

    print("\n🎤 Habla… (I'll reply when you pause)")
    t0 = time.monotonic()

    async def feed_audio():
        # Ring-buffer collector: only START a turn after PADDING_MS of mostly-voiced
        # frames, then END it after SILENCE_MS of quiet. A frame counts as speech only
        # if webrtcvad flags it AND it clears RMS_FLOOR (rejects low-level noise the
        # VAD misclassifies). A turn is never ended before it has actually started.
        window = deque(maxlen=max(1, PADDING_MS // VAD_FRAME_MS))
        triggered = False
        silence_ms = 0
        utterance_ms = 0
        if preroll:
            for f in preroll:
                frames.append(f)
                await ts_stream.input_stream.send_audio_event(audio_chunk=f)
            triggered = True
            utterance_ms = len(preroll) * VAD_FRAME_MS
        while True:
            data = await reader.q.get()
            frames.append(data)
            await ts_stream.input_stream.send_audio_event(audio_chunk=data)

            rms = _frame_rms(data)
            try:
                speech = (len(data) == FRAME_BYTES
                          and rms >= RMS_FLOOR
                          and vad.is_speech(data, RATE))
            except Exception:
                speech = False

            stats["n"] += 1
            stats["voiced"] += 1 if speech else 0
            stats["maxrms"] = max(stats["maxrms"], rms)
            if DEBUG_VAD:
                print("|" if speech else ".", end="", flush=True)

            if not triggered:
                window.append(speech)
                if sum(window) >= TRIGGER_RATIO * window.maxlen:
                    triggered = True
                    if DEBUG_VAD:
                        print("<start>", end="", flush=True)
            else:
                utterance_ms += VAD_FRAME_MS
                silence_ms = 0 if speech else silence_ms + VAD_FRAME_MS
                if silence_ms >= SILENCE_MS or utterance_ms >= MAX_UTTERANCE_MS:
                    break
        await ts_stream.input_stream.end_stream()

    await feed_audio()
    await handler.handle_events()

    transcript = handler.transcript.strip()
    wall = time.monotonic() - t0
    print(f"  [diag] wall={wall:.1f}s frames={stats['n']} voiced={stats['voiced']} "
          f"maxrms={stats['maxrms']:.0f} chars={len(transcript)}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # paInt16 = 16-bit = 2 bytes
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return transcript, buf.getvalue()


def upload_audio(api_url, audio_wav):
    info = requests.get(f"{api_url}/upload-url").json()
    requests.put(info["upload_url"], data=audio_wav,
                 headers={"Content-Type": "audio/wav"})
    return info["key"]


async def converse_turn(ws_url, text, audio_key, history, player, interrupt):
    """
    Stream the reply over the WebSocket. Returns the (possibly partial) reply text.
    Stops cleanly when `interrupt` is set — we never cancel this task from outside,
    because cancelling through `async with websockets.connect(...)` mid-receive tears
    through the close handshake and raises. Instead we race each recv against the
    interrupt event and exit the context manager normally.
    """
    reply = ""
    printed = False
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            await ws.send(json.dumps({
                "action": "converse", "text": text,
                "audio_key": audio_key, "history": history,
            }))
            recv_task = asyncio.ensure_future(ws.recv())
            interrupt_task = asyncio.ensure_future(interrupt.wait())
            try:
                while True:
                    await asyncio.wait({recv_task, interrupt_task},
                                       return_when=asyncio.FIRST_COMPLETED)
                    if interrupt_task.done():
                        break
                    try:
                        raw = recv_task.result()
                    except Exception:
                        break  # connection closed / error
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "reply_delta":
                        if not printed:
                            print("  Bot: ", end="", flush=True)
                            printed = True
                        print(m["text"], end="", flush=True)
                        reply += m["text"]
                    elif t == "audio":
                        player.enqueue(base64.b64decode(m["data"]))
                    elif t == "pronunciation_note":
                        print(f"\n  📝 {m['text']}")
                    elif t == "grammar_note":
                        print(f"\n  ✏️  {m['text']}")
                    elif t == "error":
                        print(f"\n  ❌ {m['message']}")
                    elif t == "done":
                        break
                    recv_task = asyncio.ensure_future(ws.recv())
            finally:
                for tk in (recv_task, interrupt_task):
                    if not tk.done():
                        tk.cancel()
                await asyncio.gather(recv_task, interrupt_task, return_exceptions=True)
    except Exception:
        pass  # never let a transport hiccup crash the turn
    if printed:
        print()
    return reply


async def detect_barge(reader):
    """
    Drain mic frames while the bot is speaking and return the onset frames (preroll)
    as soon as the user interrupts. Adaptive: `floor` tracks the running
    bleed/ambient RMS (updated only on non-speech frames), and a frame counts as an
    interrupt only if it's clearly louder than that floor AND flagged as speech,
    sustained across the window. Frames arriving after the trigger stay in the queue
    for the next capture, so nothing is lost.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    window = deque(maxlen=max(1, BARGE_PADDING_MS // VAD_FRAME_MS))
    recent = deque(maxlen=max(1, PREROLL_MS // VAD_FRAME_MS))
    floor = float(BARGE_RMS_MIN)
    if DEBUG_VAD:
        print("  [barge] watching…", flush=True)
    while True:
        data = await reader.q.get()
        recent.append(data)
        rms = _frame_rms(data)
        threshold = max(BARGE_RMS_MIN, floor * BARGE_FACTOR)
        try:
            loud = len(data) == FRAME_BYTES and rms >= threshold
            sp = loud and vad.is_speech(data, RATE)
        except Exception:
            sp = False
        if not sp:
            floor = 0.95 * floor + 0.05 * rms
        window.append(sp)
        if DEBUG_VAD and rms >= threshold:
            print(f"  [barge] rms={rms:.0f} thr={threshold:.0f} voiced={sum(window)}/{window.maxlen}", flush=True)
        if sum(window) >= BARGE_TRIGGER_RATIO * window.maxlen:
            if DEBUG_VAD:
                print("  [barge] INTERRUPT", flush=True)
            return list(recent)


async def play_reply_with_barge(ws_url, text, audio_key, history, player, reader):
    """
    Stream the bot's reply while watching for an interrupt. Returns
    (reply_text, barged, preroll). On barge, playback is killed, the reply task is
    cancelled, and the interrupt onset is returned for the next capture.
    """
    player.reset()
    interrupt = asyncio.Event()
    ws_task = asyncio.create_task(
        converse_turn(ws_url, text, audio_key, history, player, interrupt))
    barge_task = asyncio.create_task(detect_barge(reader))

    reply = ""
    barged = False
    preroll = None
    try:
        while True:
            if barge_task.done():
                barged = True
                preroll = barge_task.result()
                player.stop()
                interrupt.set()       # tell converse_turn to stop cleanly
                break
            if ws_task.done() and not player.is_busy():
                break
            await asyncio.sleep(0.03)
    finally:
        interrupt.set()               # ensure the reply task always unblocks
        try:
            reply = await ws_task     # returns the (partial) reply, no cancellation
        except Exception:
            reply = ""
        if not barge_task.done():
            barge_task.cancel()
            try:
                await barge_task
            except BaseException:
                pass

    if barged:
        print("\n  ⏹️  (interrupted)")
    return reply, barged, preroll


async def main():
    print("🇪🇸 ¡Hola! Let's chat in Spanish (streaming, no Enter needed).")
    print("=" * 48)
    print("Just speak — I'll reply as soon as you pause.")
    print("You can talk over me to interrupt. Say 'salir' or Ctrl+C to quit.\n")

    outputs = get_stack_outputs()
    api_url = outputs["ApiUrl"]
    ws_url = outputs["WsUrl"]

    loop = asyncio.get_event_loop()
    reader = MicReader(loop)
    player = AudioPlayer()
    history = []
    pending_preroll = None

    try:
        while True:
            text, audio_wav = await capture(reader, pending_preroll)
            pending_preroll = None
            if not text:
                print("  (no speech detected)")
                continue
            if text.lower().strip() in ("quit", "salir"):
                break

            print(f"  Tú: {text}")
            try:
                audio_key = upload_audio(api_url, audio_wav)
                reply, barged, preroll = await play_reply_with_barge(
                    ws_url, text, audio_key, history, player, reader)
            except Exception as e:
                print(f"  ❌ turn error: {e}")
                continue
            if barged:
                pending_preroll = preroll

            history.append({"role": "user", "content": text})
            if reply:
                history.append({"role": "assistant", "content": reply})
            if len(history) > 20:
                history = history[-20:]
    finally:
        reader.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: (print("\n👋 ¡Hasta luego!"), os._exit(0)))
    asyncio.run(main())
