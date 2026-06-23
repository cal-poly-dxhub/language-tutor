"""
SageMaker container for Wav2Vec2 phoneme extraction.

- GET /ping: health check
- POST /invocations: single-shot (JSON with audio_b64 + text)
- WebSocket on port 8081: streaming mode (client streams audio chunks, sends text, gets scored)
"""

import asyncio
import base64
import io
import json
import logging
import threading
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np
import soundfile as sf
import torch
import websockets
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model.eval()

# Phonemizer
espeak_backend = EspeakBackend("es", with_stress=False)
phoneme_separator = Separator(phone=" ", word=" | ")

TARGET_SR = 16000


def extract_phonemes(audio_bytes: bytes) -> list[str]:
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)

    inputs = processor(audio_data, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return [p for p in transcription.split() if p]


def phonemize_text(text: str) -> list[str]:
    result = espeak_backend.phonemize([text], separator=phoneme_separator)[0]
    return [p for p in result.replace("|", "").split() if p]


# --- HTTP server for SageMaker health checks and single-shot ---

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class SageMakerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        if self.path == "/invocations":
            try:
                data = json.loads(body)
                audio_bytes = base64.b64decode(data["audio_b64"])
                actual = extract_phonemes(audio_bytes)
                expected = phonemize_text(data["text"])
                distance = levenshtein(expected, actual)
                max_len = max(len(expected), len(actual), 1)
                accuracy = max(0, (max_len - distance) / max_len) * 100
                errors = []
                for i in range(min(len(expected), len(actual))):
                    if expected[i] != actual[i]:
                        errors.append({"position": i, "expected": expected[i], "got": actual[i]})
                result = json.dumps({
                    "actual_phonemes": actual,
                    "expected_phonemes": expected,
                    "score": {"accuracy": round(accuracy, 1), "distance": distance, "errors": errors},
                })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(result.encode())
            except Exception as e:
                logger.error(f"Error: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


# --- WebSocket server for streaming ---

async def handle_stream(websocket):
    """
    Client protocol:
    - Send binary frames: raw PCM audio chunks (16kHz, 16-bit, mono)
    - Send text frame: JSON {"text": "detected transcript"}
    - Server responds with JSON score once text is received
    """
    audio_chunks = []
    detected_text = None

    async for message in websocket:
        if isinstance(message, bytes):
            audio_chunks.append(message)
        else:
            data = json.loads(message)
            if "text" in data:
                detected_text = data["text"]
                break

    if not detected_text or not audio_chunks:
        await websocket.send(json.dumps({"error": "Missing audio or text"}))
        return

    # Assemble WAV from raw PCM chunks
    pcm = b"".join(audio_chunks)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm)

    try:
        actual = extract_phonemes(buf.getvalue())
        expected = phonemize_text(detected_text)

        # Score
        distance = levenshtein(expected, actual)
        max_len = max(len(expected), len(actual), 1)
        accuracy = max(0, (max_len - distance) / max_len) * 100
        errors = []
        for i in range(min(len(expected), len(actual))):
            if expected[i] != actual[i]:
                errors.append({"position": i, "expected": expected[i], "got": actual[i]})

        await websocket.send(json.dumps({
            "detected_text": detected_text,
            "expected_phonemes": expected,
            "actual_phonemes": actual,
            "score": {"accuracy": round(accuracy, 1), "distance": distance, "errors": errors},
        }))
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))


def levenshtein(s1: list, s2: list) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


async def ws_server():
    logger.info("Starting WebSocket server on port 8081")
    async with websockets.serve(handle_stream, "0.0.0.0", 8081):
        await asyncio.Future()


def run_ws():
    asyncio.run(ws_server())


def main():
    # Start WebSocket server in background thread
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # HTTP server on port 8080 (SageMaker requirement)
    logger.info("Starting HTTP server on port 8080")
    server = ThreadedHTTPServer(("0.0.0.0", 8080), SageMakerHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
