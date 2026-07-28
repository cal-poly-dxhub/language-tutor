"""
Tests for the asynchronous coach.

The coach is where "feedback only when relevant" is enforced, so these tests focus on
the decisions rather than the AWS plumbing: which turns are worth reviewing, whether a
model response actually contains a note, silence trimming, and the guarantee that a
pronunciation claim is never made without phoneme evidence.
"""

import array
import asyncio
import json
import math
import os
import sys
import wave

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bridge"))

import coach as C  # noqa: E402


# --- helpers -----------------------------------------------------------------

def tone(ms, amplitude=8000, rate=C.SAMPLE_RATE):
    n = int(rate * ms / 1000)
    samples = array.array(
        "h", (int(amplitude * math.sin(2 * math.pi * 220 * i / rate)) for i in range(n)))
    return samples.tobytes()


def silence(ms, rate=C.SAMPLE_RATE):
    return b"\x00\x00" * int(rate * ms / 1000)


# --- audio helpers -----------------------------------------------------------

def test_frame_rms_distinguishes_silence_from_speech():
    assert C.frame_rms(silence(50)) == 0.0
    assert C.frame_rms(tone(50)) > C.SILENCE_RMS_FLOOR


def test_frame_rms_handles_odd_length_and_empty():
    assert C.frame_rms(b"") == 0.0
    assert C.frame_rms(b"\x01") == 0.0


def test_trim_silence_keeps_the_voiced_region():
    pcm = silence(500) + tone(400) + silence(500)
    trimmed = C.trim_silence(pcm)
    # Voiced region plus at most the configured padding on each side.
    assert 400 <= C.duration_ms(trimmed) <= 400 + 2 * C.TRIM_PAD_MS + 40
    assert C.duration_ms(pcm) > 1300


def test_trim_silence_returns_empty_for_pure_silence():
    # Silence must not reach Wav2Vec2: it would invent phonemes nobody spoke.
    assert C.trim_silence(silence(1000)) == b""


def test_pcm_to_wav_roundtrips():
    pcm = tone(100)
    with wave.open(__import__("io").BytesIO(C.pcm_to_wav(pcm))) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == C.SAMPLE_RATE
        assert wf.readframes(wf.getnframes()) == pcm


# --- language neutrality ------------------------------------------------------
# The coach deliberately contains no vocabulary of any language. These tests pin that
# down: nothing is filtered on the basis of which language a turn is in, or how short
# it is. The reviewer model decides, and its "no note" answer is what produces silence.

def test_module_contains_no_language_specific_word_lists():
    source = open(C.__file__, encoding="utf-8").read()
    # Any of these appearing would mean a language had leaked back into the code.
    for token in ("Spanish", "español", "seseo", "yeísmo", "hola", "gracias",
                  "_ES_MARKERS", "_EN_MARKERS", "_FILLERS", "looks_spanish"):
        assert token not in source, f"{token!r} should not appear in coach.py"


def make_coach():
    return C.Coach(sagemaker_endpoint="ep", model_id="m", system_prompt="p",
                   sagemaker_client=object(), bedrock_client=object())


@pytest.mark.parametrize("text", [
    "yo comí una manzana ayer",          # target language, long
    "perro",                             # target language, one word
    "sí",                                # backchannel
    "what is the homework for tonight",  # learner's own language
    "je suis allé au marché",            # a different target language entirely
    "私は学生です",                        # a different script entirely
])
def test_every_utterance_with_words_reaches_the_reviewer(text):
    assert make_coach().should_analyze(text, tone(600)) is True


@pytest.mark.parametrize("text", ["", "   ", "...", "123", "!!!"])
def test_only_wordless_turns_are_skipped(text):
    # Nothing to review and nothing to send anywhere.
    assert make_coach().should_analyze(text, tone(600)) is False


# --- note parsing ------------------------------------------------------------

def test_parse_notes_extracts_both():
    raw = '{"pronunciation": "your rr sounded like a tap", "grammar": "use comí"}'
    assert C.parse_notes(raw) == {
        "pronunciation": "your rr sounded like a tap", "grammar": "use comí"}


def test_parse_notes_tolerates_code_fences_and_prose():
    raw = 'Here you go:\n```json\n{"pronunciation": null, "grammar": "use fui"}\n```'
    assert C.parse_notes(raw) == {"grammar": "use fui"}


@pytest.mark.parametrize("raw", [
    '{"pronunciation": null, "grammar": null}',
    '{"pronunciation": "", "grammar": "none"}',
    '{"pronunciation": "N/A", "grammar": "  "}',
    '{"grammar": "nothing"}',
])
def test_parse_notes_treats_nullish_as_no_note(raw):
    assert C.parse_notes(raw) == {}


@pytest.mark.parametrize("raw", ["", "not json at all", "[]", "null", "{oops"])
def test_parse_notes_survives_garbage(raw):
    assert C.parse_notes(raw) == {}


def test_parse_notes_ignores_unknown_keys():
    assert C.parse_notes('{"vocabulary": "x", "grammar": "y"}') == {"grammar": "y"}


# --- dedupe ------------------------------------------------------------------

def test_deduper_suppresses_repeats_across_accents_and_case():
    d = C.NoteDeduper()
    assert d.accept("Your 'rr' sounded soft") is True
    assert d.accept("your rr sounded soft") is False
    assert d.accept("YOUR RR SOUNDED SOFT!") is False
    assert d.accept("Different note entirely") is True


def test_deduper_rejects_blank():
    assert C.NoteDeduper().accept("   ") is False


# --- coach input -------------------------------------------------------------

def test_build_coach_input_marks_missing_phonemes():
    payload = C.build_coach_input("hola amigo", [], [], None)
    assert "unavailable" in payload
    assert "grammar only" in payload


def test_build_coach_input_includes_phonemes_and_accuracy():
    payload = C.build_coach_input("hola", ["o", "l", "a"], ["o", "l", "a"], 97.5)
    assert "Expected phonemes: o l a" in payload
    assert "97.5%" in payload


def test_build_coach_input_caps_phoneme_length():
    long_seq = ["a"] * (C.MAX_PHONEMES + 500)
    payload = C.build_coach_input("x", long_seq, long_seq, 50.0)
    # Identical sequences, so the only "a"s are the two capped phoneme lines.
    assert payload.count("a") <= 2 * C.MAX_PHONEMES + 60


# --- end to end with stubbed AWS --------------------------------------------

class FakeBody:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload


class FakeSageMaker:
    def __init__(self, expected, actual, accuracy):
        self.payload = {"expected_phonemes": expected, "actual_phonemes": actual,
                        "score": {"accuracy": accuracy}}
        self.calls = 0

    def invoke_endpoint(self, **kwargs):
        self.calls += 1
        return {"Body": FakeBody(self.payload)}


class FakeBedrock:
    def __init__(self, text):
        self.text = text
        self.calls = 0
        self.last_body = None

    def invoke_model(self, **kwargs):
        self.calls += 1
        self.last_body = json.loads(kwargs["body"])
        return {"body": FakeBody({"content": [{"type": "text", "text": self.text}]})}


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


def test_analyze_returns_both_notes_when_phonemes_available():
    sm = FakeSageMaker(["p", "e", "r", "r", "o"], ["p", "e", "r", "o"], 78.0)
    br = FakeBedrock('{"pronunciation": "rr came out as a tap", "grammar": "use tengo"}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    notes = review(c, "yo tener un perro", tone(1200))
    assert notes == [("pronunciation", "rr came out as a tap"),
                     ("grammar", "use tengo")]
    assert sm.calls == 1 and br.calls == 1


def test_analyze_is_silent_when_nothing_is_wrong():
    sm = FakeSageMaker(["o", "l", "a"], ["o", "l", "a"], 99.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "hola cómo estás", tone(1200)) == []


def test_reviewer_decides_silence_for_native_language_questions():
    # No local language guess: the turn goes to Haiku, which returns nothing for a
    # question asked in the learner's own language. Silence comes from judgement, not
    # from a keyword list.
    sm = FakeSageMaker(["a"], ["b"], 40.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "what is the homework due date", tone(1200)) == []
    assert br.calls == 1


def test_wordless_turn_reaches_no_aws_service():
    sm = FakeSageMaker([], [], None)
    br = FakeBedrock('{"grammar": "should not happen"}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "   ", tone(1200)) == []
    assert sm.calls == 0 and br.calls == 0


def test_pronunciation_note_dropped_without_phoneme_evidence():
    # No SageMaker endpoint configured (e.g. -c phonemes=false): the model must not
    # be allowed to assert pronunciation problems it cannot have observed.
    br = FakeBedrock('{"pronunciation": "guessed from spelling", "grammar": "use fui"}')
    c = C.Coach(sagemaker_endpoint="", bedrock_client=br)
    assert review(c, "ayer yo ir al cine", tone(1200)) == [("grammar", "use fui")]


def test_pronunciation_skipped_when_audio_is_all_silence():
    sm = FakeSageMaker(["a"], ["b"], 10.0)
    br = FakeBedrock('{"pronunciation": "p", "grammar": "g"}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    notes = review(c, "ayer yo ir al cine", silence(2000))
    assert notes == [("grammar", "g")]
    assert sm.calls == 0


def test_analyze_dedupes_across_turns():
    sm = FakeSageMaker(["a"], ["b"], 60.0)
    br = FakeBedrock('{"grammar": "conjugate the verb"}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "ayer yo ir al cine", tone(1200)) == [
        ("grammar", "conjugate the verb")]
    assert review(c, "mañana yo ir al cine", tone(1200)) == []


def test_analyze_survives_sagemaker_failure():
    class Boom:
        def invoke_endpoint(self, **kwargs):
            raise RuntimeError("endpoint down")

    br = FakeBedrock('{"grammar": "use fui"}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=Boom(), bedrock_client=br)
    # Grammar still lands; pronunciation is suppressed for lack of evidence.
    assert review(c, "ayer yo ir al cine", tone(1200)) == [("grammar", "use fui")]


def test_analyze_survives_bedrock_failure():
    class Boom:
        def invoke_model(self, **kwargs):
            raise RuntimeError("throttled")

    sm = FakeSageMaker(["a"], ["b"], 60.0)
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=Boom())
    assert review(c, "ayer yo ir al cine", tone(1200)) == []


def test_coach_sends_transcript_and_prompt_to_the_model():
    sm = FakeSageMaker(["o", "l", "a"], ["o", "l", "a"], 95.0)
    br = FakeBedrock('{"grammar": null, "pronunciation": null}')
    c = C.Coach(sagemaker_endpoint="ep", system_prompt="COACH RULES",
                sagemaker_client=sm, bedrock_client=br)
    review(c, "hola cómo estás", tone(1200))
    assert br.last_body["system"] == "COACH RULES"
    assert "hola cómo estás" in br.last_body["messages"][0]["content"]
    assert br.last_body["temperature"] == 0.0


# --- diagnostics -------------------------------------------------------------
# "Nothing to fix" has several very different causes. These pin down that the coach
# records which one applied, so a silent coach can be told apart from a healthy one.

def test_diagnostics_report_a_missing_endpoint():
    br = FakeBedrock('{"pronunciation": "guessed", "grammar": null}')
    c = C.Coach(sagemaker_endpoint="", bedrock_client=br)
    assert review(c, "ayer yo ir al cine", tone(1200)) == []
    assert "no SAGEMAKER_ENDPOINT" in c.diagnostics["phonemeSkip"]
    assert c.diagnostics["dropped"] == ["pronunciation:no-phoneme-evidence"]


def test_diagnostics_report_audio_below_the_silence_floor():
    sm = FakeSageMaker(["a"], ["b"], 50.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    review(c, "ayer yo ir al cine", silence(2000))
    assert "silence floor" in c.diagnostics["phonemeSkip"]
    assert sm.calls == 0


def test_diagnostics_report_an_endpoint_failure():
    class Boom:
        def invoke_endpoint(self, **kwargs):
            raise RuntimeError("endpoint down")

    br = FakeBedrock('{"grammar": null, "pronunciation": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=Boom(), bedrock_client=br)
    review(c, "ayer yo ir al cine", tone(1200))
    assert "endpoint down" in c.diagnostics["phonemeSkip"]


def test_diagnostics_are_clean_when_pronunciation_was_scored():
    sm = FakeSageMaker(["p", "e", "r", "r", "o"], ["p", "e", "r", "o"], 78.0)
    br = FakeBedrock('{"pronunciation": "rr came out as a tap", "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    notes = review(c, "yo tengo un perro", tone(1500))
    assert notes == [("pronunciation", "rr came out as a tap")]
    assert c.diagnostics["phonemeSkip"] is None
    assert c.diagnostics["phonemes"] == 4
    assert c.diagnostics["accuracy"] == 78.0
    assert c.diagnostics["dropped"] == []


def test_diagnostics_report_a_suppressed_repeat():
    sm = FakeSageMaker(["a"], ["b"], 60.0)
    br = FakeBedrock('{"grammar": "conjugate the verb"}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    review(c, "ayer yo ir al cine", tone(1200))
    review(c, "mañana yo ir al cine", tone(1200))
    assert c.diagnostics["dropped"] == ["grammar:repeat"]


def test_pronunciation_dropped_when_accuracy_implies_a_wrong_transcript():
    """
    An accuracy near zero means the transcript is not what the learner said — typically
    their own language transcribed into the target language. Coaching the phonemes of
    words they never uttered produces confidently wrong advice.
    """
    sm = FakeSageMaker(["k", "w", "e", "n", "t", "a"], ["t", "e", "l", "m", "i"], 11.1)
    br = FakeBedrock('{"pronunciation": "say cuéntame slowly", "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "cuéntame sobre el silabo", tone(1500)) == []
    assert any("transcript-unreliable" in d for d in c.diagnostics["dropped"])


def test_pronunciation_kept_when_accuracy_is_merely_poor():
    # Genuinely bad pronunciation of the right words must still be flagged.
    sm = FakeSageMaker(["o", "l", "a", "k", "o", "m", "o"], ["o", "l", "a", "k", "a", "m", "u"], 58.3)
    br = FakeBedrock('{"pronunciation": "your vowels shifted", "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "hola cómo estás", tone(1500)) == [
        ("pronunciation", "your vowels shifted")]
    assert c.diagnostics["dropped"] == []


# --- phoneme differences -----------------------------------------------------
# Handing the reviewer two flat phoneme lists made it compute the alignment itself and
# invent sounds — it once reported "the rr in tortillas", a word with no rr. The diff is
# computed here so the prompt can forbid claims that are not in it.

def test_alignment_finds_a_substitution():
    ops = C.align_phonemes(list("olakomo"), list("olakamo"))
    assert len(ops) == 1
    assert ops[0]["expected"] == ["o"] and ops[0]["heard"] == ["a"]
    assert ops[0]["kind"] == "replaced"


def test_alignment_finds_an_inserted_sound():
    ops = C.align_phonemes(list("ola"), list("hola"))
    assert ops[0]["kind"] == "added" and ops[0]["heard"] == ["h"]


def test_alignment_finds_a_missing_sound():
    ops = C.align_phonemes(list("hola"), list("ola"))
    assert ops[0]["kind"] == "missing" and ops[0]["expected"] == ["h"]


def test_identical_sequences_have_no_differences():
    assert C.align_phonemes(list("ola"), list("ola")) == []
    assert "No phoneme differences" in C.describe_differences([])


# --- the reviewer judges accent, not code ------------------------------------
# Classifying substitutions in code was tried and removed: a table of phoneme pairs cannot
# weigh position, neighbouring sounds or dialect, and an LLM can. The diff is computed as
# evidence, but every judgement about it belongs to the reviewer.

def test_all_differences_are_shown_to_the_reviewer_unfiltered():
    payload = C.build_coach_input(
        "tortillas", ["t", "i", "ʝ", "a"], ["t", "i", "ʃ", "a"], 95.0)
    assert "expected ʝ but heard ʃ" in payload
    assert "judge each one yourself" in payload


def test_the_reviewer_may_decline_a_difference():
    sm = FakeSageMaker(["t", "i", "ʝ", "a"], ["t", "i", "ʃ", "a"], 95.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "tortillas", tone(1500)) == []
    assert c.diagnostics["dropped"] == []


def test_the_reviewers_note_is_passed_through():
    sm = FakeSageMaker(["t", "i", "ʝ", "a"], ["t", "i", "l", "a"], 95.0)
    br = FakeBedrock('{"pronunciation": "the ll should sound like a y", "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "tortillas", tone(1500)) == [
        ("pronunciation", "the ll should sound like a y")]


def test_differences_are_recorded_in_the_diagnostics():
    sm = FakeSageMaker(["o", "l", "a"], ["h", "o", "l", "a"], 93.0)
    br = FakeBedrock('{"pronunciation": null, "grammar": null}')
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    review(c, "hola", tone(1500))
    assert any("heard h" in line for line in c.diagnostics["differences"])


# --- word attribution --------------------------------------------------------
# Expected phonemes come from the Transcribe transcript, word by word, so every
# difference can be mapped to the word it falls in. Handing over a flat list instead made
# the reviewer guess, and it guessed confidently: one note told a learner about "the LL in
# discriminatorio", a word containing no LL.

WORDS = [
    {"word": "me", "phonemes": ["m", "e"]},
    {"word": "gusta", "phonemes": ["ɣ", "u", "s", "t", "a"]},
    {"word": "comer", "phonemes": ["k", "o", "m", "e", "ɾ"]},
    {"word": "tortillas", "phonemes": ["t", "o", "ɾ", "t", "i", "ʝ", "a", "s"]},
]
FLAT = [p for w in WORDS for p in w["phonemes"]]


def _with(index, value):
    actual = FLAT.copy()
    actual[index] = value
    return actual


def test_difference_is_attributed_to_the_right_word():
    ops = C.align_phonemes(FLAT, _with(11, "t"))       # the ɾ of "comer"
    C.attribute_to_words(ops, WORDS)
    assert ops[0]["word"] == "comer"
    assert 'in the word "comer"' in C.describe_differences(ops)


def test_difference_in_a_later_word_is_attributed_correctly():
    ops = C.align_phonemes(FLAT, _with(17, "l"))       # the ʝ of "tortillas"
    C.attribute_to_words(ops, WORDS)
    assert ops[0]["word"] == "tortillas"


def test_inserted_sound_is_attributed_to_the_word_it_precedes():
    ops = C.align_phonemes(FLAT, ["h"] + FLAT)
    C.attribute_to_words(ops, WORDS)
    assert ops[0]["word"] == "me"


def test_no_word_grouping_leaves_differences_unattributed():
    # An endpoint built before word grouping was added: say nothing rather than guess.
    ops = C.align_phonemes(FLAT, _with(11, "t"))
    C.attribute_to_words(ops, [])
    assert "in the word" not in C.describe_differences(ops)


def test_word_list_reaches_the_reviewer():
    payload = C.build_coach_input("me gusta comer tortillas", FLAT, _with(11, "t"),
                                 95.0, WORDS)
    assert 'in the word "comer"' in payload
    assert "do not infer a different one" in payload


# --- fabricated citations ----------------------------------------------------

def test_a_note_citing_a_word_never_said_is_detected():
    note = "In 'discriminatorio', your 'LL' came out as a plain 'L' sound"
    assert C.cites_a_word_not_said(note, "me gusta comer tortillas") == "discriminatorio"


def test_a_note_citing_a_real_word_is_accepted():
    note = "In 'tortillas', the 'll' should have a y-like sound, not a plain 'l'"
    assert C.cites_a_word_not_said(note, "me gusta comer tortillas") is None


def test_short_quoted_sounds_are_not_treated_as_citations():
    note = "Your 'rr' came out as 'r' — keep it rolled"
    assert C.cites_a_word_not_said(note, "el perro corre") is None


def test_fabricated_note_is_dropped_by_the_coach():
    sm = FakeSageMaker(["t", "i", "ʝ", "a"], ["t", "i", "l", "a"], 90.0)
    import json as _json
    br = FakeBedrock(_json.dumps({
        "pronunciation": "In 'discriminatorio' your ll became an l",
        "grammar": None}))
    c = C.Coach(sagemaker_endpoint="ep", sagemaker_client=sm, bedrock_client=br)
    assert review(c, "me gusta comer tortillas", tone(1500)) == []
    assert any("cited-word-not-said:discriminatorio" in d
               for d in c.diagnostics["dropped"])
