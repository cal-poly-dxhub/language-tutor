"""
Asynchronous pronunciation + grammar coach.

This is the "slow lane" of the tutor. The conversation model owns the conversation and
must never wait on us: it hears the learner and answers in ~1s. Meanwhile, for each
finished learner turn, the bridge hands us (transcript, raw PCM) and forgets about it.
We take as long as we need (typically 1-3s), and if — and only if — the reviewer finds
something worth saying, we hand back one or two short notes that the bridge pushes to
the client as on-screen annotations. The spoken conversation is never interrupted.

Pipeline per turn:
    1. Wav2Vec2 on SageMaker -> expected vs. actual phoneme sequences + accuracy.
       (This is the project's differentiator; the speech-to-speech model gives no
       phoneme detail.)
    2. The reviewer model    -> decides what actually matters, returns strict JSON.
    3. Dedupe                -> the same note is never shown twice in a session.

LANGUAGE NEUTRALITY
This module contains no words from any language, and no heuristics about any language.
Deciding whether a turn is worth commenting on — whether it is the target language, a
question in the learner's own language, a backchannel, or a genuine mistake — is the
reviewer model's job, and it is far better at that than a keyword list would be. All
language-specific content lives in `languages/*.json` and the prompt templates.

What is enforced here is only what the model cannot know or should not be trusted with:
    - a blank transcript has nothing to review, so nothing is sent anywhere;
    - near-silent audio is never sent to the phoneme scorer, which would otherwise
      invent phonemes nobody spoke;
    - a pronunciation note is impossible without phoneme evidence, so one is dropped if
      the audio was never scored — the model cannot infer sound from spelling;
    - the same note is not repeated.

Everything here is synchronous/pure except `Coach.analyze`, which pushes the blocking
AWS calls onto a thread so the bridge's audio pump is never stalled.
"""

from __future__ import annotations

import array
import asyncio
import base64
import io
import json
import logging
import os
import re
import unicodedata
import wave
from collections import deque
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

# Audio handling thresholds. These are properties of the signal, not of a language.
MIN_SPEECH_MS = 400            # too short to score phonemes reliably
SILENCE_RMS_FLOOR = 300        # 0-32767; below this a frame is treated as silence
TRIM_PAD_MS = 120              # keep a little context around detected speech
MAX_PHONEMES = 200             # cap prompt size for very long turns
# Below this phoneme accuracy the transcript is almost certainly not what the learner
# said — most often because they spoke their own language and the model transcribed it
# into the target language. Expected phonemes derived from a wrong transcript describe
# words nobody uttered, so a pronunciation note built on them is not just unhelpful, it
# is false. The prompt says this too; this enforces it.
MIN_TRUSTWORTHY_ACCURACY = 35.0
DEDUPE_HISTORY = 24            # remember this many notes per session

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
# Ways a model may spell "I have no note". Compared case-insensitively; these are
# JSON/English protocol tokens from the prompt contract, not target-language content.
_NULLISH = {"", "none", "null", "n/a", "na", "nothing", "no notes", "-", "—"}

DEFAULT_COACH_PROMPT = (
    "You review one completed utterance from a language learner and decide whether "
    "anything is worth telling them. Return strict JSON only: "
    '{"pronunciation": <note or null>, "grammar": <note or null>}'
)


# --- pure helpers (unit-tested) ----------------------------------------------

def frame_rms(pcm: bytes) -> float:
    """RMS loudness of little-endian 16-bit mono PCM, on a 0-32767 scale."""
    usable = len(pcm) - (len(pcm) % SAMPLE_WIDTH)
    if usable <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[:usable])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def trim_silence(pcm: bytes, rate: int = SAMPLE_RATE,
                 floor: float = SILENCE_RMS_FLOOR,
                 pad_ms: int = TRIM_PAD_MS,
                 frame_ms: int = 20) -> bytes:
    """
    Drop leading/trailing silence from a turn buffer.

    The mic streams continuously, so a turn buffer usually starts and ends with room
    noise (and possibly bleed from the tutor's own voice). Feeding that to Wav2Vec2
    invents phonemes that were never spoken, which would produce bogus pronunciation
    notes. Trimming to the voiced region keeps scoring honest.
    """
    frame_bytes = max(SAMPLE_WIDTH, (rate * frame_ms // 1000) * SAMPLE_WIDTH)
    frames = [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    voiced = [i for i, f in enumerate(frames) if frame_rms(f) >= floor]
    if not voiced:
        return b""
    pad = max(0, pad_ms // frame_ms)
    start = max(0, voiced[0] - pad)
    end = min(len(frames), voiced[-1] + 1 + pad)
    return b"".join(frames[start:end])


def duration_ms(pcm: bytes, rate: int = SAMPLE_RATE) -> float:
    return (len(pcm) / SAMPLE_WIDTH) / rate * 1000.0


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 mono in a WAV container (what the SageMaker model expects)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def words(text: str) -> List[str]:
    """Script-agnostic word split, used only to test for 'any word at all'."""
    return _WORD_RE.findall(text or "")


def normalize_note(text: str) -> str:
    """Canonical form used for dedupe: accent/case/punctuation insensitive."""
    stripped = unicodedata.normalize("NFKD", (text or "").lower())
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(_WORD_RE.findall(stripped))


def is_nullish(value) -> bool:
    """The reviewer may signal 'nothing to report' as null, '', 'none', etc."""
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _NULLISH


def parse_notes(raw: str) -> dict:
    """
    Pull {"pronunciation": ..., "grammar": ...} out of a model response.

    Tolerates code fences and surrounding prose. Returns only the keys that carry real
    content, so an empty dict means "nothing worth saying" — the common case.
    """
    if not raw:
        return {}
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        brace = text.find("{")
        if brace == -1:
            return {}
        text = text[brace:]
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[:i + 1]
                    break
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for key in ("pronunciation", "grammar"):
        value = data.get(key)
        if not is_nullish(value) and isinstance(value, str):
            out[key] = value.strip()
    return out


def align_phonemes(expected: List[str], actual: List[str], limit: int = 12) -> List[dict]:
    """
    Diff the expected phoneme sequence against what was actually heard.

    This exists because handing the reviewer two flat lists made it compute the alignment
    itself, and it did so badly — inventing sounds that were not in the word at all (it
    once reported "the 'rr' in tortillas", which contains no rr). Computing the diff here
    turns a reasoning task into a lookup, and lets the prompt forbid mentioning any sound
    that is not in this list.

    Returns [{"expected": [...], "heard": [...], "kind": "replaced|missing|added"}, ...].
    """
    import difflib

    ops = []
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        exp, heard = expected[i1:i2], actual[j1:j2]
        kind = {"replace": "replaced", "delete": "missing", "insert": "added"}[tag]
        ops.append({"expected": exp, "heard": heard, "kind": kind,
                    "at": i1})
    # Longest differences first: those carry the most information.
    ops.sort(key=lambda o: -(len(o["expected"]) + len(o["heard"])))
    return ops[:limit]


def attribute_to_words(ops: List[dict], expected_words: Optional[List[dict]]) -> None:
    """
    Tag each difference with the word it falls in, using the word-grouped expected
    phonemes from the phonemiser.

    Without this the reviewer receives a flat phoneme diff and has to guess which word a
    sound belongs to. It guesses wrong and confidently: one note told a learner about
    "the LL in discriminatorio", a word with no LL in it. The expected phonemes come from
    the Transcribe transcript word by word, so the mapping is arithmetic, not inference.
    """
    if not expected_words:
        return
    # Absolute start index of each word within the flat expected sequence.
    spans, cursor = [], 0
    for entry in expected_words:
        length = len(entry.get("phonemes") or [])
        spans.append((cursor, cursor + length, entry.get("word", "")))
        cursor += length
    for op in ops:
        at = op.get("at", 0)
        word = ""
        for start, end, name in spans:
            # An insertion sits at a boundary; attribute it to the word it precedes.
            if start <= at < end or (op["kind"] == "added" and at == start):
                word = name
                break
        if not word and spans and at >= spans[-1][1]:
            word = spans[-1][2]
        op["word"] = word


def describe_differences(ops: List[dict]) -> str:
    """Render the diff as lines the reviewer can quote from without inference."""
    if not ops:
        return "No phoneme differences: what they said matched the expected sounds."
    lines = []
    for op in ops:
        exp = " ".join(op["expected"]) or "(nothing)"
        heard = " ".join(op["heard"]) or "(nothing)"
        where = f" in the word \"{op['word']}\"" if op.get("word") else ""
        if op["kind"] == "added":
            lines.append(f"- added a sound that should not be there{where}: "
                         f"heard {heard}")
        elif op["kind"] == "missing":
            lines.append(f"- expected {exp}{where} but nothing was heard there")
        else:
            lines.append(f"- expected {exp} but heard {heard}{where}")
    return "\n".join(lines)


def cites_a_word_not_said(note: str, transcript: str) -> Optional[str]:
    """
    Return the first quoted word in `note` that the learner did not actually say.

    A verifiable check, not a judgement: the reviewer quotes words in single or double
    quotes, and any quoted run of letters that is absent from the transcript is a
    fabrication. This is the guard that would have caught "the LL in discriminatorio"
    being attached to a word the learner never uttered.
    """
    said = {w.lower() for w in words(transcript)}
    if not said:
        return None
    for quoted in re.findall(r"['\"“”‘’]([^'\"“”‘’]{2,40})['\"“”‘’]", note):
        tokens = words(quoted)
        # Ignore quotes that are explanations rather than citations (e.g. "y-like"),
        # and single letters or letter pairs being described.
        if len(tokens) != 1 or len(tokens[0]) < 4:
            continue
        if tokens[0].lower() not in said:
            return tokens[0]
    return None


def build_coach_input(transcript: str, expected: List[str], actual: List[str],
                      accuracy: Optional[float],
                      expected_words: Optional[List[dict]] = None) -> str:
    """
    The user-turn payload handed to the reviewer model.

    The phoneme difference list is computed here and handed over as evidence, but it is
    NOT filtered: judging which differences are regional accent and which are mistakes is
    the reviewer's job. An LLM can weigh position, neighbouring sounds and dialect
    together, which a table of substitution pairs cannot. Computing the diff simply stops
    it having to align two sequences by eye — that is where it used to invent sounds that
    were not in the word.
    """
    lines = [f'Learner said (speech recognition): "{transcript}"']
    if expected or actual:
        lines.append(f"Expected phonemes: {' '.join(expected[:MAX_PHONEMES])}")
        lines.append(f"Actual phonemes:   {' '.join(actual[:MAX_PHONEMES])}")
        lines.append("")
        ops = align_phonemes(expected, actual)
        attribute_to_words(ops, expected_words)
        lines.append("Differences found (computed, complete — judge each one yourself). "
                     "The word each one falls in is given; use that word, do not infer a "
                     "different one:")
        lines.append(describe_differences(ops))
    else:
        lines.append("Phoneme data: unavailable (grammar only — do not comment on "
                     "pronunciation).")
    if accuracy is not None:
        lines.append(f"Phoneme accuracy: {accuracy:.1f}%")
    return "\n".join(lines)


class NoteDeduper:
    """Suppresses repeats so the learner isn't told the same thing every turn."""

    def __init__(self, size: int = DEDUPE_HISTORY):
        self._seen = deque(maxlen=size)

    def accept(self, text: str) -> bool:
        key = normalize_note(text)
        if not key or key in self._seen:
            return False
        self._seen.append(key)
        return True


# --- the coach ---------------------------------------------------------------

class Coach:
    """
    Per-session coach. `analyze` is the only entry point the bridge calls; it offloads
    blocking AWS calls to a thread so the event loop keeps pumping audio.
    """

    def __init__(self, sagemaker_endpoint: str = "", model_id: str = "",
                 system_prompt: str = "", region: str = "",
                 espeak_language: str = "",
                 sagemaker_client=None, bedrock_client=None):
        self.sagemaker_endpoint = sagemaker_endpoint or os.environ.get("SAGEMAKER_ENDPOINT", "")
        self.model_id = model_id or os.environ.get(
            "COACH_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.system_prompt = system_prompt or os.environ.get(
            "COACH_PROMPT", DEFAULT_COACH_PROMPT)
        self.region = region or os.environ.get("BEDROCK_REGION", "us-west-2")
        # espeak voice for phonemising the transcript. Must match the language being
        # taught, or "expected" describes the wrong sounds entirely.
        self.espeak_language = espeak_language or os.environ.get("ESPEAK_LANGUAGE", "")
        self._sagemaker = sagemaker_client
        self._bedrock = bedrock_client
        self.deduper = NoteDeduper()
        # NOTE: no per-turn state on this object. Two utterances can be in flight at
        # once, and a shared "last result" field would let the later one overwrite the
        # earlier one's report. Per-call data is returned, not stored.

    # Clients are created lazily so this module imports (and unit-tests) with no AWS
    # credentials present.
    def _sagemaker_client(self):
        if self._sagemaker is None:
            import boto3
            self._sagemaker = boto3.client("sagemaker-runtime", region_name=self.region)
        return self._sagemaker

    def _bedrock_client(self):
        if self._bedrock is None:
            import boto3
            self._bedrock = boto3.client("bedrock-runtime", region_name=self.region)
        return self._bedrock

    # ---- gate ----------------------------------------------------------------
    def should_analyze(self, transcript: str, pcm: bytes) -> bool:
        """
        The only reason to skip a turn outright: there are no words in it.

        Everything else — which language it was in, whether it was a one-word
        acknowledgement, whether the grammar is fine — is a judgement call, and the
        reviewer model makes it. Guessing here with keyword lists would be both
        language-specific and worse, and a wrong guess silently costs the learner
        feedback they should have received.
        """
        return bool(words(transcript))

    # ---- blocking stages (run in a worker thread) ---------------------------
    def _phonemes(self, transcript: str, pcm: bytes):
        """Returns (expected, actual, accuracy, expected_words)."""
        self.phoneme_skip = None
        if not self.sagemaker_endpoint:
            self.phoneme_skip = "no SAGEMAKER_ENDPOINT configured"
            return [], [], None, []
        if not pcm:
            self.phoneme_skip = "no audio was teed for this turn"
            return [], [], None, []
        speech = trim_silence(pcm)
        if duration_ms(speech) < MIN_SPEECH_MS:
            self.phoneme_skip = (
                f"only {duration_ms(speech):.0f}ms of audio cleared the silence floor "
                f"(RMS {frame_rms(pcm):.0f} < {SILENCE_RMS_FLOOR}); needs "
                f"{MIN_SPEECH_MS}ms")
            return [], [], None, []
        try:
            resp = self._sagemaker_client().invoke_endpoint(
                EndpointName=self.sagemaker_endpoint,
                ContentType="application/json",
                Body=json.dumps({
                    "audio_b64": base64.b64encode(pcm_to_wav(speech)).decode(),
                    "text": transcript,
                    "language": self.espeak_language,
                }))
            result = json.loads(resp["Body"].read())

            # VERIFY THE ENDPOINT HONOURED THE LANGUAGE. An endpoint built before
            # per-language phonemisation reads every language with one fixed voice, and a
            # request for a different one is answered with confident nonsense: a greeting
            # read with the wrong voice measured 14% against a correct recording of it.
            # Those phonemes are not weak evidence, they describe a different word — and
            # believing that score suppresses the whole turn, so the learner gets silence
            # in every language except the one the endpoint was built for. Discarding them
            # instead leaves grammar coaching working while the endpoint is stale.
            served = (result.get("language") or "").strip()
            wanted = (self.espeak_language or "").strip()
            if wanted and served != wanted:
                self.phoneme_skip = (
                    f"endpoint phonemised in {served or 'an unreported language'}, not "
                    f"{wanted} — redeploy the phoneme container to score this language")
                logger.warning("phoneme endpoint served %r for a %r request; "
                               "discarding the score", served or None, wanted)
                return [], [], None, []

            score = result.get("score") or {}
            return (result.get("expected_phonemes") or [],
                    result.get("actual_phonemes") or [],
                    score.get("accuracy"),
                    # Absent on an endpoint built before word grouping was added; the
                    # differences then simply carry no word, which is honest.
                    result.get("expected_words") or [])
        except Exception as exc:  # noqa: BLE001 - coaching must never break the call
            self.phoneme_skip = f"{type(exc).__name__}: {exc}"
            logger.warning("phoneme extraction failed: %s", exc)
            return [], [], None, []

    def _evaluate(self, payload: str) -> dict:
        try:
            resp = self._bedrock_client().invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 400,
                    "temperature": 0.0,
                    "system": self.system_prompt,
                    "messages": [{"role": "user", "content": payload}],
                }))
            body = json.loads(resp["body"].read())
            text = "".join(b.get("text", "") for b in body.get("content", [])
                           if b.get("type") == "text")
            return parse_notes(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("coach evaluation failed: %s", exc)
            return {}

    def _analyze_blocking(self, transcript: str,
                          pcm: bytes) -> List[Tuple[str, str]]:
        # THE CONVERSATION TRANSCRIPT IS THE EVIDENCE.
        #
        # "Expected" phonemes must describe what the learner was TRYING to say, and the
        # conversation model's transcript does that: it infers the words they were reaching
        # for, so a mispronunciation shows up as a difference between expected and actual.
        # A second recogniser reading the audio literally was tried here and removed — it
        # wrote what it heard, so phonemising it made expected equal actual and the error
        # cancelled out, and it manufactured grammar errors out of its own mishearings.
        evidence = transcript
        trimmed = trim_silence(pcm) if pcm else b""
        expected, actual, accuracy, expected_words = self._phonemes(evidence, pcm)
        differences = align_phonemes(expected, actual)
        attribute_to_words(differences, expected_words)
        raw_notes = self._evaluate(
            build_coach_input(evidence, expected, actual, accuracy, expected_words))

        # Why the coach reached its verdict. "Nothing to fix" has several very different
        # causes — no phoneme evidence, audio trimmed to nothing, the reviewer declining,
        # a note suppressed as a repeat — and they are indistinguishable from outside.
        diagnostics = {
            "audioMs": round(duration_ms(pcm)) if pcm else 0,
            "speechMs": round(duration_ms(trimmed)),
            "rms": round(frame_rms(pcm)) if pcm else 0,
            "endpoint": bool(self.sagemaker_endpoint),
            "phonemes": len(actual),
            "accuracy": accuracy,
            "reviewer": sorted(raw_notes),
            "phonemeSkip": getattr(self, "phoneme_skip", None),
            "differences": describe_differences(differences).splitlines(),
            "wordAligned": bool(expected_words),
            "dropped": [],
        }
        if diagnostics["phonemeSkip"]:
            logger.info("no phoneme evidence: %s", diagnostics["phonemeSkip"])

        # A LOW PHONEME SCORE SUPPRESSES PRONUNCIATION ONLY, never grammar.
        #
        # Accuracy is a measurement of SOUNDS, and it drops for two unrelated reasons: the
        # transcript describes different words, or the alignment is simply poor — short
        # utterances, a heavy accent, a clipped recording. Grammar does not depend on any
        # of that. It depends on the transcript, which the conversation model gets right.
        #
        # Making a low score silence the whole turn was tried and reverted: it removed
        # grammar coaching from exactly the learners whose audio aligns worst, and a real
        # error in a short sentence went unflagged because the phonemes scored badly. The
        # wrong-words case it was meant to catch does not need this: a transcript in
        # another language reads as correct prose, so the reviewer returns nothing anyway,
        # and it is told explicitly to ignore turns in the learner's own language.
        #
        # The suppression that remains is narrow and provable: a claim ABOUT SOUNDS needs
        # sound evidence, so pronunciation is dropped below the floor. See the loop below.

        # NO LANGUAGE GATE, deliberately. Deciding the spoken language from recogniser
        # confidence was tried and removed: a learner speaking the target language WITH
        # AN ACCENT frequently scores higher under their own language, and that is the
        # entire population this tool serves. Measured on real learner speech, a turn
        # transcribed correctly in the target language was labelled as the native one, and
        # the gate discarded a turn that had a real grammar note waiting.
        #
        # Judging the language from the TEXT is the reviewer's job and it does it well:
        # the prompt tells it to return nothing for a turn in the learner's own language.
        out = []
        for category in ("pronunciation", "grammar"):
            text = raw_notes.get(category)
            if not text:
                continue
            # Pronunciation claims require phoneme evidence; without it the model would
            # be guessing sound from spelling.
            if category == "pronunciation" and not actual:
                diagnostics["dropped"].append(f"{category}:no-phoneme-evidence")
                continue
            # And the evidence has to be worth something. Below the floor the alignment
            # says the sounds do not correspond to these words at all, so a claim about
            # WHICH sound was wrong is not supported. Grammar is untouched by this: it
            # rests on the transcript, not on the audio.
            if (category == "pronunciation" and accuracy is not None
                    and accuracy < MIN_TRUSTWORTHY_ACCURACY):
                diagnostics["dropped"].append(
                    f"{category}:accuracy-{accuracy:.0f}%-below-"
                    f"{MIN_TRUSTWORTHY_ACCURACY:.0f}%-no-reliable-sound-evidence")
                continue
            # A note that quotes a word the learner never said is worse than no note:
            # it is confidently wrong and it destroys trust in every other note.
            fabricated = cites_a_word_not_said(text, evidence)
            if fabricated:
                diagnostics["dropped"].append(
                    f"{category}:cited-word-not-said:{fabricated}")
                logger.warning("dropped a %s note citing %r, absent from %r",
                               category, fabricated, evidence[:60])
                continue
            if not self.deduper.accept(text):
                diagnostics["dropped"].append(f"{category}:repeat")
                continue
            out.append((category, text))
        return out, diagnostics

    # ---- async entry point --------------------------------------------------
    async def analyze(self, transcript: str, pcm: bytes):
        """
        Returns (notes, diagnostics). `notes` is [(category, note), ...] and is usually
        empty. Both belong to THIS call: nothing is stored on the Coach, so concurrent
        utterances cannot overwrite each other's results. Never raises — a failure to
        coach must not disturb the conversation.

        """
        if not self.should_analyze(transcript, pcm):
            return [], {"skipped": "no words in the transcript"}
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._analyze_blocking, transcript, pcm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("coach failed: %s", exc)
            return [], {"error": str(exc)}
