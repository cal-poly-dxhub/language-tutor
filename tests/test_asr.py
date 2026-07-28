"""
Tests for the independent coaching-lane ASR.

The contract: Amazon Transcribe reads the learner's audio with no conversational context,
so it supplies (a) a transcript that has not been translated into the language being
taught, and (b) the language actually spoken. Nova Sonic can provide neither reliably —
once a session is primed in Spanish it renders English speech as Spanish words, which
makes phoneme scoring meaningless and pronunciation notes false.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bridge"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import asr as A  # noqa: E402
import coach as C  # noqa: E402


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def review(coach, *args):
    """
    Run one review and return just the notes.

    `analyze` returns (notes, diagnostics) so that concurrent utterances cannot overwrite
    each other's report — there is deliberately no per-turn state on the Coach. Tests that
    inspect the report read `coach.diagnostics`, which this helper attaches for
    convenience.
    """
    notes, diagnostics = run(coach.analyze(*args))
    coach.diagnostics = diagnostics
    return notes


# --- language comparison -----------------------------------------------------

@pytest.mark.parametrize("spoken,target,expected", [
    ("es-US", "es-US", True),
    ("es-ES", "es-US", True),      # dialect is never the question
    ("en-US", "es-US", False),
    ("fr-FR", "fr-FR", True),
    ("en-US", "en-US", True),
    ("hi-IN", "en-US", False),
])
def test_is_target_language(spoken, target, expected):
    assert A.is_target_language(spoken, target) is expected


def test_unknown_language_defers_to_the_reviewer():
    # Identification unavailable, or audio too short to judge: do not silently drop the
    # turn, let the model decide as it does everywhere else.
    assert A.is_target_language(None, "es-US") is True
    assert A.is_target_language("", "es-US") is True


# --- Transcription ------------------------------------------------------------

def test_transcription_ok_requires_text_and_no_error():
    assert A.Transcription("hola amigo", "es-US").ok is True
    assert A.Transcription("", "es-US").ok is False
    assert A.Transcription("hola", "es-US", error="boom").ok is False


def test_transcription_strips_whitespace():
    assert A.Transcription("  hola amigo  ").text == "hola amigo"


# --- sidecar client ----------------------------------------------------------
# Transcribe runs in its own container because amazon-transcribe pins awscrt ~=0.26.1
# while the Nova Sonic SDK needs ~=0.28.2 — no released pair resolves. The bridge reaches
# it over localhost.

class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []

    def post(self, url, params=None, data=None, timeout=None, headers=None):
        self.calls.append({"url": url, "params": params, "bytes": len(data or b"")})
        if self.raises:
            raise self.raises
        return self.response


def test_transcript_and_language_are_returned():
    session = FakeSession(FakeResponse(200, {"text": "hola cómo estás",
                                             "language": "es-US"}))
    result = run(A.transcribe(b"\x01\x02" * 800, ["es-US", "en-US"], session=session))
    assert result.text == "hola cómo estás"
    assert result.language == "es-US"
    assert result.ok is True
    assert session.calls[0]["params"] == {"languages": "es-US,en-US"}
    assert session.calls[0]["bytes"] == 1600


def test_sidecar_error_is_reported_not_raised():
    session = FakeSession(FakeResponse(200, {"text": "", "language": None,
                                            "error": "transcribe unavailable"}))
    result = run(A.transcribe(b"\x01\x02" * 800, ["es-US", "en-US"], session=session))
    assert result.ok is False
    assert result.error == "transcribe unavailable"


def test_unreachable_sidecar_degrades_gracefully():
    session = FakeSession(raises=OSError("connection refused"))
    result = run(A.transcribe(b"\x01\x02" * 800, ["es-US", "en-US"], session=session))
    assert result.ok is False
    assert "connection refused" in result.error


def test_non_200_from_the_sidecar_is_an_error():
    session = FakeSession(FakeResponse(500, None, "boom"))
    result = run(A.transcribe(b"\x01\x02" * 800, ["es-US", "en-US"], session=session))
    assert result.ok is False
    assert "500" in result.error


def test_no_audio_or_no_languages_short_circuits():
    session = FakeSession(FakeResponse(200, {"text": "x"}))
    assert run(A.transcribe(b"", ["es-US", "en-US"], session=session)).ok is False
    assert run(A.transcribe(b"\x00\x00", [], session=session)).ok is False
    assert session.calls == [], "must not call the sidecar with nothing to do"


# --- the coach uses it -------------------------------------------------------

class FakeBody:
    def __init__(self, payload):
        import json
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload


class FakeSageMaker:
    def __init__(self, expected, actual, accuracy):
        self.payload = {"expected_phonemes": expected, "actual_phonemes": actual,
                        "score": {"accuracy": accuracy}}
        self.texts = []

    def invoke_endpoint(self, **kwargs):
        import json
        self.texts.append(json.loads(kwargs["Body"])["text"])
        return {"Body": FakeBody(self.payload)}


class FakeBedrock:
    def __init__(self, text):
        self.text = text
        self.last_body = None

    def invoke_model(self, **kwargs):
        import json
        self.last_body = json.loads(kwargs["body"])
        return {"body": FakeBody({"content": [{"type": "text", "text": self.text}]})}


def tone(ms, rate=16000):
    import array, math
    n = int(rate * ms / 1000)
    return array.array("h", (int(8000 * math.sin(2 * math.pi * 220 * i / rate))
                             for i in range(n))).tobytes()


def test_phonemes_are_scored_against_the_independent_transcript():
    """
    The heart of it. Nova Sonic mistranscribed the learner into the target language;
    scoring phonemes against those words would describe sounds nobody made.
    """
    sm = FakeSageMaker(["o", "l", "a"], ["o", "l", "a"], 96.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", target_language_code="es-US",
                sagemaker_client=sm, bedrock_client=br)
    asr = A.Transcription("hola cómo estás", "es-US")
    review(c, "cuéntame sobre el silabo", tone(1200), asr)
    assert sm.texts == ["hola cómo estás"], "Sonic's transcript must not be the evidence"
    assert "hola cómo estás" in br.last_body["messages"][0]["content"]
    assert c.diagnostics["usedAsr"] is True
    assert c.diagnostics["asrLanguage"] == "es-US"


def test_accented_target_language_is_still_graded():
    """
    The regression that motivated removing the language gate. A learner speaking the
    target language with a heavy accent is frequently mis-identified as speaking their own
    language. Gating on that discarded turns that had real notes waiting — the opposite of
    what a pronunciation tutor should do.
    """
    sm = FakeSageMaker(["o", "l", "a"], ["h", "o", "l", "a"], 50.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": "estoy, not esta"}')
    c = C.Coach(sagemaker_endpoint="ep", target_language_code="es-US",
                sagemaker_client=sm, bedrock_client=br)
    asr = A.Transcription("Hola yo esta bien.", "en-US")   # mis-identified on purpose
    assert review(c, "Hola yo esta bien.", tone(3200), asr) == [
        ("grammar", "estoy, not esta")]
    assert c.diagnostics["dropped"] == []


def test_a_reported_language_never_discards_a_turn():
    sm = FakeSageMaker(["a"], ["b"], 80.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": "fix the verb"}')
    c = C.Coach(sagemaker_endpoint="ep", target_language_code="es-US",
                sagemaker_client=sm, bedrock_client=br)
    for language in ("en-US", "es-US", None, "fr-FR"):
        notes = review(c, "una frase de prueba", tone(1500),
                       A.Transcription("una frase de prueba", language))
        assert not any("spoken-language" in d for d in c.diagnostics["dropped"]), language
        c.deduper = C.NoteDeduper()      # so the repeat guard does not mask the result
        assert notes == [("grammar", "fix the verb")], language


def test_dialect_of_the_target_language_is_still_graded():
    sm = FakeSageMaker(["o", "l", "a"], ["o", "l", "e"], 60.0)
    br = FakeBedrock('{"pronunciation": "watch the final vowel", "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", target_language_code="es-US",
                sagemaker_client=sm, bedrock_client=br)
    asr = A.Transcription("hola cómo estás", "es-ES")
    assert review(c, "hola cómo estás", tone(1200), asr) == [
        ("pronunciation", "watch the final vowel")]


def test_sonic_transcript_is_used_when_asr_failed():
    sm = FakeSageMaker(["o", "l", "a"], ["o", "l", "a"], 90.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", target_language_code="es-US",
                sagemaker_client=sm, bedrock_client=br)
    asr = A.Transcription("", None, error="transcribe unavailable")
    review(c, "hola amigo", tone(1200), asr)
    assert sm.texts == ["hola amigo"]
    assert c.diagnostics["usedAsr"] is False
    assert c.diagnostics["asrError"] == "transcribe unavailable"


def test_coaching_still_works_with_no_asr_at_all():
    sm = FakeSageMaker(["o"], ["a"], 55.0)
    br = FakeBedrock('{"pronunciation": "vowel drifted", "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "hola amigo", tone(1200)) == [("pronunciation", "vowel drifted")]
    assert c.diagnostics["usedAsr"] is False


# --- language identification contract ----------------------------------------
# Verified against the live service: Transcribe's own streaming identification returns an
# EMPTY transcript for audio under roughly four seconds, which would lose most learner
# turns. The sidecar instead transcribes each candidate language concurrently with a fixed
# language_code and picks the higher mean word confidence, then withholds the claim when
# that confidence is weak. Measured outcomes:
#     0.5s English -> en-US (0.997)
#     0.4s Spanish -> undecided (0.657, and the winner was wrong)
#     1.4s English -> en-US (0.996)
#     3.0s Spanish -> es-US (0.903)

def test_a_missing_language_claim_changes_nothing():
    """Language is advisory: present, absent or wrong, the turn is reviewed either way."""
    sm = FakeSageMaker(["o", "l", "a"], ["o", "l", "a"], 88.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": "use fui"}')
    c = C.Coach(sagemaker_endpoint="ep", target_language_code="es-US",
                sagemaker_client=sm, bedrock_client=br)
    asr = A.Transcription("hola", None)          # transcript, no language claim
    assert review(c, "hola", tone(1200), asr) == [("grammar", "use fui")]
    assert not any("spoken-language" in d for d in c.diagnostics["dropped"])




# --- correcting the on-screen transcript -------------------------------------
# The sidecar returns two readings: `text` in the target language (which drives phoneme
# scoring) and `display`, the best reading of what was actually said. Showing the target
# reading of native-language speech was a regression — "yo what's good cuh" came back as
# "good" and replaced a correct transcript with a mangled one.

def test_display_and_scoring_transcripts_are_separate():
    t = A.Transcription("good", "es-US", None, "yo what's good cuh", "en-US")
    assert t.text == "good"                    # what the phoneme scorer sees
    assert t.display == "yo what's good cuh"   # what the learner sees
    assert t.detected == "en-US"


def test_display_defaults_to_the_transcript():
    assert A.Transcription("hola amigo", "es-US").display == "hola amigo"


@pytest.mark.parametrize("sonic,asr,expected", [
    ("yo what's good cuh", "good", False),           # the regression: mangled reading
    ("hola qué tal", "hola cómo estás", True),       # genuine disagreement, same length
    ("cuéntame sobre el silabo", "tell me about the syllabus", True),
    ("hola amigo", "hola amigo", False),             # identical, nothing to correct
    ("hola amigo", "Hola, amigo.", False),           # punctuation only
    ("algo que decir", "", False),                   # nothing to offer
])
def test_only_plausible_replacements_are_shown(sonic, asr, expected):
    assert C.worth_correcting(sonic, asr) is expected
