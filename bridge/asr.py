"""
Client for the coaching lane's independent ASR (the Transcribe sidecar).

WHY A SIDECAR AND NOT A LIBRARY CALL
`amazon-transcribe` pins `awscrt ~= 0.26.1`; the Nova Sonic SDK's `smithy-http[awscrt]`
pins `awscrt ~= 0.28.2`. No released version pair resolves, so Transcribe runs in its own
container beside this one and is reached over localhost. See asr_service/app.py.

WHY AN INDEPENDENT ASR AT ALL
Nova Sonic's transcript is conditioned on the conversation: once the tutor has spoken the
target language, speech in the learner's own language comes back rendered as target
language words. Harmless for the conversation, fatal for coaching — Wav2Vec2 scores the
learner's sounds against the phonemisation of that transcript, so a translated transcript
yields expected phonemes for words nobody said, and every note built on it is false.

Transcribe hears only the audio, so it has no reason to translate, and language
identification tells us which language was actually spoken. That replaces guesswork
(keyword lists, accuracy thresholds) with evidence.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
ASR_URL = os.environ.get("ASR_URL", "http://127.0.0.1:8081/transcribe")
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "12"))


class Transcription:
    """What one utterance's independent ASR produced."""

    def __init__(self, text: str = "", language: Optional[str] = None,
                 error: Optional[str] = None, display: str = "",
                 detected: Optional[str] = None):
        # `text` is the target-language reading, used for phoneme scoring. `display` is
        # the best reading of what was actually said, used for the on-screen transcript.
        self.text = (text or "").strip()
        self.display = (display or text or "").strip()
        self.language = language
        self.detected = detected
        self.error = error

    @property
    def ok(self) -> bool:
        return bool(self.text) and self.error is None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Transcription(text={self.text!r}, language={self.language!r}, "
                f"error={self.error!r})")


def is_target_language(language: Optional[str], target_code: str) -> bool:
    """
    True when the utterance was identified as the language being taught.

    Compares the language subtag only, so es-US and es-ES both count as Spanish: the
    tutor teaches a language, and dialect is precisely what the coach must never flag. An
    unknown language (identification unavailable, or audio too short to judge) counts as
    the target language, deferring to the reviewer model rather than silently dropping the
    turn.
    """
    if not language:
        return True
    return language.split("-")[0].lower() == (target_code or "").split("-")[0].lower()


async def transcribe(pcm: bytes, languages: List[str], session=None,
                     url: str = "", timeout: float = 0.0) -> Transcription:
    """
    Ask the sidecar to transcribe one utterance.

    Never raises: a Transcription carrying an error is always returned, because losing
    ASR must degrade coaching and nothing else.
    """
    if not pcm:
        return Transcription(error="no audio for this turn")
    if not languages:
        return Transcription(error="no language options configured")

    import aiohttp

    url = url or ASR_URL
    timeout = timeout or ASR_TIMEOUT
    params = {"languages": ",".join(languages)}
    try:
        if session is not None:
            resp_json = await _post(session, url, params, pcm, timeout)
        else:
            async with aiohttp.ClientSession() as own:
                resp_json = await _post(own, url, params, pcm, timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("asr sidecar unreachable: %s", exc)
        return Transcription(error=f"{type(exc).__name__}: {exc}")

    return Transcription(resp_json.get("text", ""), resp_json.get("language"),
                         resp_json.get("error"), resp_json.get("display", ""),
                         resp_json.get("detected"))


async def _post(session, url: str, params: dict, pcm: bytes, timeout: float) -> dict:
    import aiohttp

    async with session.post(url, params=params, data=pcm,
                            timeout=aiohttp.ClientTimeout(total=timeout),
                            headers={"Content-Type": "application/octet-stream"}) as r:
        if r.status != 200:
            body = (await r.text())[:200]
            return {"error": f"sidecar returned {r.status}: {body}"}
        return await r.json()
