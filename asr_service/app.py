"""
Transcribe sidecar: an independent ASR for the coaching lane.

WHY THIS IS A SEPARATE CONTAINER
It would be simpler to call Amazon Transcribe from the bridge process, but the two SDKs
cannot share a Python environment:

    amazon-transcribe 0.6.4      depends on awscrt ~= 0.26.1
    smithy-http[awscrt] 0.2.x    depends on awscrt ~= 0.28.2   (Nova Sonic's SDK)

Every version pair of the two was tried and pip resolves none of them. So Transcribe
lives here, in its own dependency set, as a second container in the same Fargate task.
The bridge reaches it over localhost, so there is no network hop and no cold start.

WHY AN INDEPENDENT ASR AT ALL
Nova Sonic returns a transcript, but it is conditioned on the conversation: once the
tutor has spoken Spanish, English speech comes back rendered as Spanish words. That is
harmless for the conversation and fatal for coaching, because Wav2Vec2 scores the
learner's sounds against the phonemisation of that transcript — words they never said
produce expected phonemes they were never trying to make. Transcribe hears only the
audio, so it has no reason to translate, and with language identification it also reports
which language was actually spoken. That turns "was this the target language?" from a
guess into evidence.

API
    GET  /health
    POST /transcribe?languages=es-US,en-US
         body: raw PCM16 mono 16kHz
         ->   {"text": "...", "language": "es-US"}          on success
              {"text": "", "language": null, "error": "…"}  on failure, with status 200
         Failure is reported in the body, not as an HTTP error: losing ASR must degrade
         coaching, never break the conversation.
"""

import asyncio
import logging
import os

from aiohttp import web
from amazon_transcribe.client import TranscribeStreamingClient

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("asr")

REGION = os.environ.get("AWS_REGION") or os.environ.get("BEDROCK_REGION", "us-west-2")
SAMPLE_RATE = 16000
# ~100ms per chunk. Transcribe wants a steady stream rather than one huge frame.
CHUNK_BYTES = SAMPLE_RATE * 2 // 10
MAX_BODY = 64 * 1024 * 1024
RESULT_TIMEOUT = float(os.environ.get("ASR_RESULT_TIMEOUT", "10"))


async def transcribe(pcm: bytes, languages: list) -> dict:
    """
    Transcribe one utterance. `languages[0]` is the language being taught.

    The returned transcript is ALWAYS the target-language reading, because that is what
    the phoneme scorer needs: expected phonemes are derived from this text, so it must be
    spelled in the language the learner is practising.

    Detection is advisory only. It is reported for diagnostics and never used to discard a
    turn: a learner speaking the target language with a strong accent frequently scores
    higher under their native language, so acting on it silences coaching for exactly the
    people who need it. Measured on real speech, "Hola yo esta bien" transcribed correctly
    as Spanish yet scored higher as en-US.
    """
    target = await _stream_once(pcm, languages[0])
    # "text" is the target-language reading and drives phoneme scoring. "display" is the
    # best reading of what they actually said, for the on-screen transcript: forcing the
    # target language onto native-language speech mangles it ("yo what's good cuh" came
    # back as "good"), and showing that as a correction is worse than showing nothing.
    result = {"text": target["text"], "display": target["text"],
              "language": languages[0], "confidence": target["confidence"]}

    others = [c for c in languages[1:]]
    if not others or not target["text"]:
        result["detected"] = languages[0] if target["text"] else None
        return result

    # One extra reading per alternative language, for the diagnostic only.
    alternates = await asyncio.gather(*(_stream_once(pcm, c) for c in others),
                                     return_exceptions=True)
    scored = [target] + [a for a in alternates if isinstance(a, dict) and a.get("text")]
    best = max(scored, key=lambda c: (c["confidence"], len(c["text"])))
    result["detected"] = best["language"]
    result["display"] = best["text"]
    result["alternatives"] = [
        {"language": c["language"], "confidence": round(c["confidence"], 3),
         "text": c["text"][:60]} for c in scored if c is not target]
    logger.info("target=%s conf=%.3f detected=%s (advisory)",
                languages[0], target["confidence"], result["detected"])
    return result


async def _stream_once(pcm: bytes, language_code: str) -> dict:
    """One Transcribe streaming session over the whole utterance, in one language."""
    client = TranscribeStreamingClient(region=REGION)
    stream = await client.start_stream_transcription(
        language_code=language_code, media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm")

    parts, confidences = [], []

    async def collect():
        async for event in stream.output_stream:
            results = getattr(getattr(event, "transcript", None), "results", None)
            for result in results or []:
                if result.is_partial:
                    continue           # partials get revised; they would duplicate
                for alt in result.alternatives or []:
                    if alt.transcript:
                        parts.append(alt.transcript)
                    for item in alt.items or []:
                        # Punctuation carries no confidence signal about the language.
                        if item.confidence is not None and item.item_type != "punctuation":
                            confidences.append(item.confidence)

    reader = asyncio.create_task(collect())
    for offset in range(0, len(pcm), CHUNK_BYTES):
        await stream.input_stream.send_audio_event(
            audio_chunk=pcm[offset:offset + CHUNK_BYTES])
    await stream.input_stream.end_stream()
    try:
        await asyncio.wait_for(reader, timeout=RESULT_TIMEOUT)
    except asyncio.TimeoutError:
        reader.cancel()
        logger.warning("no final result within %ss for %s", RESULT_TIMEOUT, language_code)

    text = " ".join(p.strip() for p in parts if p.strip()).strip()
    return {"text": text, "language": language_code,
            "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0}


async def handle_transcribe(request):
    languages = [c.strip() for c in
                 (request.query.get("languages") or "").split(",") if c.strip()]
    if not languages:
        return web.json_response({"text": "", "language": None,
                                  "error": "no languages requested"})
    pcm = await request.read()
    if not pcm:
        return web.json_response({"text": "", "language": None, "error": "empty body"})
    try:
        result = await transcribe(pcm, languages)
        logger.info("%.2fs audio -> lang=%s text=%r",
                    len(pcm) / (SAMPLE_RATE * 2), result["language"],
                    result["text"][:80])
        return web.json_response(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("transcribe failed: %s", exc)
        return web.json_response({"text": "", "language": None,
                                  "error": f"{type(exc).__name__}: {exc}"})


async def health(_request):
    return web.Response(text="ok")


def make_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BODY)
    app.router.add_get("/health", health)
    app.router.add_post("/transcribe", handle_transcribe)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "8081")))
