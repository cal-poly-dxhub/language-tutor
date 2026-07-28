"""
Tests for the bridge's turn-boundary rule.

The contract these lock down: the coach is handed the WHOLE thing the learner said in
one go — long or short — and never a fragment. Nova Sonic delivers the learner's ASR
transcript as one or more TEXT content blocks; blocks that close with
stop reasons are NOT a usable end-of-turn signal — verified against the live model, a
complete utterance still closes its ASR block with `PARTIAL_TURN`. The authoritative
boundary is the `userSpeechEnd` event. Also covered: the audio tee lines up with the
transcript, and barge-in is surfaced to the client.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bridge"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import server as S  # noqa: E402


class FakeWS:
    closed = False

    def __init__(self):
        self.sent = []

    async def send_str(self, data):
        self.sent.append(data)


def make_session():
    """A session with the Nova Sonic stream and the coach replaced by recorders."""
    session = S.SonicSession(FakeWS())
    session.to_client = []
    session.dispatched = []

    async def capture(obj):
        session.to_client.append(obj)

    session._to_client = capture
    session._dispatch_coach = lambda text, asr=None: session.dispatched.append(
        (text, bytes(session._turn_pcm)))
    return session


def feed(session, *events):
    async def go():
        for event in events:
            await session._handle_event(event)
    asyncio.new_event_loop().run_until_complete(go())


def user_start(content_id="c1"):
    return {"contentStart": {"role": "USER", "type": "TEXT", "contentId": content_id,
                             "additionalModelFields": '{"generationStage":"FINAL"}'}}


def text_out(text, role="USER"):
    return {"textOutput": {"content": text, "role": role}}


def content_end(stop, content_id="c1", ctype="TEXT"):
    return {"contentEnd": {"type": ctype, "stopReason": stop, "contentId": content_id}}


def speech_start():
    return {"userSpeechStart": {"inputAudioOffsetMs": 0}}


def speech_end():
    return {"userSpeechEnd": {"inputAudioOffsetMs": 3040,
                              "inputAudioDetectionOffsetMs": 3640}}


# --- the core rule -----------------------------------------------------------
# Verified against the live model: a COMPLETE learner utterance closes its ASR content
# block with stopReason PARTIAL_TURN, and the authoritative end-of-turn signal is the
# userSpeechEnd event. Gating on END_TURN would mean the coach never ran at all.

def test_complete_utterance_dispatches_on_speech_end():
    s = make_session()
    feed(s, speech_start(), user_start(), text_out("yo comí una manzana"),
         content_end("PARTIAL_TURN"), speech_end())
    assert [t for t, _ in s.dispatched] == ["yo comí una manzana"]


def test_dispatch_works_when_speech_end_precedes_the_transcript():
    # Observed ordering varies: sometimes userSpeechEnd arrives before the ASR text.
    s = make_session()
    feed(s, speech_start(), speech_end(), user_start(),
         text_out("yo comí una manzana"), content_end("PARTIAL_TURN"))
    assert [t for t, _ in s.dispatched] == ["yo comí una manzana"]


def test_transcript_alone_does_not_dispatch():
    # Without userSpeechEnd the learner may still be talking.
    s = make_session()
    feed(s, speech_start(), user_start(), text_out("yo comí"),
         content_end("PARTIAL_TURN"))
    assert s.dispatched == []


def test_partial_turn_stop_reason_does_not_block_dispatch():
    s = make_session()
    feed(s, speech_start(), user_start(), text_out("hola amigo"),
         content_end("PARTIAL_TURN"), speech_end())
    assert [t for t, _ in s.dispatched] == ["hola amigo"]


def test_fragments_are_joined_and_dispatched_once():
    s = make_session()
    feed(s, speech_start(),
         user_start("a"), text_out("ayer fui"), content_end("PARTIAL_TURN", "a"),
         user_start("b"), text_out("al mercado"), content_end("PARTIAL_TURN", "b"),
         user_start("c"), text_out("con mi hermana"), content_end("PARTIAL_TURN", "c"),
         speech_end())
    assert [t for t, _ in s.dispatched] == ["ayer fui al mercado con mi hermana"]


def test_multiple_text_events_in_one_block_are_joined():
    s = make_session()
    feed(s, speech_start(), user_start(), text_out("hola"), text_out("cómo estás"),
         content_end("PARTIAL_TURN"), speech_end())
    assert [t for t, _ in s.dispatched] == ["hola cómo estás"]


def test_short_complete_utterance_still_dispatches():
    # "this can be long or short" — length is not a dispatch criterion.
    s = make_session()
    feed(s, speech_start(), user_start(), text_out("perro"),
         content_end("PARTIAL_TURN"), speech_end())
    assert [t for t, _ in s.dispatched] == ["perro"]


def test_consecutive_turns_dispatch_separately():
    s = make_session()
    feed(s, speech_start(), user_start("a"), text_out("hola amigo"),
         content_end("PARTIAL_TURN", "a"), speech_end())
    feed(s, speech_start(), user_start("b"), text_out("estoy bien"),
         content_end("PARTIAL_TURN", "b"), speech_end())
    assert [t for t, _ in s.dispatched] == ["hola amigo", "estoy bien"]


def test_one_utterance_is_never_dispatched_twice():
    s = make_session()
    feed(s, speech_start(), user_start(), text_out("hola amigo"),
         content_end("PARTIAL_TURN"), speech_end(),
         {"completionEnd": {"stopReason": "END_TURN"}})
    assert [t for t, _ in s.dispatched] == ["hola amigo"]


def test_completion_end_is_a_safety_net_for_leftover_text():
    # If the ASR block never closes, the turn is still unambiguously over once the
    # model has finished answering.
    s = make_session()
    feed(s, speech_start(), user_start(), text_out("hola amigo"), speech_end(),
         {"completionEnd": {"stopReason": "END_TURN"}})
    assert [t for t, _ in s.dispatched] == ["hola amigo"]


def test_assistant_text_never_reaches_the_coach():
    s = make_session()
    feed(s,
         {"contentStart": {"role": "ASSISTANT", "type": "TEXT", "contentId": "z",
                           "additionalModelFields": '{"generationStage":"SPECULATIVE"}'}},
         text_out("¡Qué bien!", role="ASSISTANT"),
         content_end("PARTIAL_TURN", "z"),
         content_end("END_TURN", "y", ctype="AUDIO"))
    assert s.dispatched == []


def test_empty_transcript_is_not_dispatched():
    s = make_session()
    s._dispatch_coach = S.SonicSession._dispatch_coach.__get__(s)  # real implementation
    feed(s, speech_start(), user_start(), text_out("   "),
         content_end("PARTIAL_TURN"), speech_end())
    # Real _dispatch_coach returns early on blank text, so no task is created.
    assert s._coach_tasks == set()


# --- audio tee ---------------------------------------------------------------

def test_audio_tee_accompanies_the_transcript_and_resets():
    s = make_session()
    s._dispatch_coach = S.SonicSession._dispatch_coach.__get__(s)
    captured = []
    s._coach_turn = lambda text, pcm, asr=None: captured.append((text, pcm)) or _noop()

    pcm = b"\x11\x22" * 800

    async def go():
        s.active = True
        s.stream = None
        s._send = _async_noop
        await s.send_audio(__import__("base64").b64encode(pcm).decode())
        assert len(s._turn_pcm) == len(pcm)
        await s._handle_event(speech_start())
        await s._handle_event(user_start())
        await s._handle_event(text_out("hola amigo"))
        await s._handle_event(content_end("PARTIAL_TURN"))
        await s._handle_event(speech_end())

    asyncio.new_event_loop().run_until_complete(go())
    assert captured and captured[0][0] == "hola amigo"
    assert captured[0][1] == pcm
    # Buffer must be cleared so the next turn is not graded on this one's audio.
    assert bytes(s._turn_pcm) == b""


def test_audio_tee_is_capped():
    s = make_session()
    oversized = b"\x01\x02" * (S.MAX_TURN_BYTES)  # twice the cap

    async def go():
        s.active = True
        s._send = _async_noop
        await s.send_audio(__import__("base64").b64encode(oversized).decode())

    asyncio.new_event_loop().run_until_complete(go())
    assert len(s._turn_pcm) == S.MAX_TURN_BYTES


def _noop():
    async def inner():
        return None
    return inner()


async def _async_noop(*args, **kwargs):
    return None


# --- client-facing events ----------------------------------------------------

def test_interrupted_stop_reason_is_forwarded():
    s = make_session()
    feed(s, {"contentEnd": {"type": "AUDIO", "stopReason": "INTERRUPTED",
                            "contentId": "x"}})
    assert {"type": "interrupted"} in s.to_client


def test_inline_interrupted_marker_is_forwarded():
    s = make_session()
    feed(s, {"textOutput": {"content": '{ "interrupted" : true }', "role": "ASSISTANT"}})
    assert {"type": "interrupted"} in s.to_client


def test_transcripts_are_forwarded_with_role_and_stage():
    s = make_session()
    feed(s, user_start(), text_out("hola amigo"))
    msg = [m for m in s.to_client if m.get("type") == "transcript"]
    assert len(msg) == 1
    assert msg[0]["role"] == "USER"
    assert msg[0]["text"] == "hola amigo"
    assert msg[0]["stage"] == "FINAL"


def test_audio_output_is_forwarded():
    s = make_session()
    feed(s, {"audioOutput": {"content": "QUJD"}})
    assert {"type": "audio", "data": "QUJD"} in s.to_client


def test_completion_end_closes_the_turn():
    s = make_session()
    feed(s, {"completionEnd": {"stopReason": "END_TURN"}})
    assert any(m.get("type") == "turn_end" for m in s.to_client)


def test_assistant_audio_end_turn_closes_the_turn():
    """
    The reliable end-of-turn signal. Observed against the live model, a turn's last
    event is the ASSISTANT audio block closing with END_TURN and completionEnd may never
    arrive — so the client would never close its speech bubble and every reply would
    pile into one.
    """
    s = make_session()
    feed(s, content_end("END_TURN", "a1", ctype="AUDIO"))
    assert any(m.get("type") == "turn_end" for m in s.to_client)


def test_partial_audio_end_does_not_close_the_turn():
    # Mid-turn audio blocks close with PARTIAL_TURN; closing there would split one
    # reply across several bubbles.
    s = make_session()
    feed(s, content_end("PARTIAL_TURN", "a1", ctype="AUDIO"))
    assert not any(m.get("type") == "turn_end" for m in s.to_client)


def test_two_replies_produce_two_turn_ends():
    s = make_session()
    feed(s, text_out("Hola", role="ASSISTANT"),
         content_end("END_TURN", "a1", ctype="AUDIO"),
         text_out("¿Y tú?", role="ASSISTANT"),
         content_end("END_TURN", "a2", ctype="AUDIO"))
    assert len([m for m in s.to_client if m.get("type") == "turn_end"]) == 2


def test_tool_use_is_captured_then_resolved(monkeypatch):
    s = make_session()
    monkeypatch.setattr(S, "retrieve_from_kb", lambda q: f"results for {q}")
    results = []

    async def fake_result(tool_id, text):
        results.append((tool_id, text))

    s._send_tool_result = fake_result
    feed(s,
         {"toolUse": {"toolName": "search_materials", "toolUseId": "t1",
                      "content": '{"query": "unidad 3"}'}},
         content_end("TOOL_USE", "t1", ctype="TOOL"))
    assert results == [("t1", "results for unidad 3")]


def test_unknown_tool_is_answered_with_an_error():
    s = make_session()
    results = []

    async def fake_result(tool_id, text):
        results.append((tool_id, text))

    s._send_tool_result = fake_result
    feed(s,
         {"toolUse": {"toolName": "log_feedback", "toolUseId": "t2", "content": "{}"}},
         content_end("TOOL_USE", "t2", ctype="TOOL"))
    assert results and "unknown tool" in results[0][1]


# --- tool result encoding ----------------------------------------------------

def test_tool_result_is_sent_as_stringified_json():
    """
    The API requires a stringified JSON object. Sending a Knowledge Base passage as raw
    prose fails the entire stream with "ValidationException: Tool Response parsing
    error", which silently broke every course-materials lookup.
    """
    s = make_session()
    sent = []
    s._send = lambda event: _record(sent, event)

    asyncio.new_event_loop().run_until_complete(
        s._send_tool_result("t1", "Week 1: greetings. Midterm in week 6."))

    payloads = [e["event"]["toolResult"]["content"] for e in sent
                if "toolResult" in e.get("event", {})]
    assert len(payloads) == 1
    decoded = __import__("json").loads(payloads[0])       # must parse as JSON
    assert decoded["result"] == "Week 1: greetings. Midterm in week 6."


def test_tool_result_that_is_already_json_is_passed_through():
    s = make_session()
    sent = []
    s._send = lambda event: _record(sent, event)

    asyncio.new_event_loop().run_until_complete(
        s._send_tool_result("t1", '{"error": "unknown tool"}'))

    payload = [e["event"]["toolResult"]["content"] for e in sent
               if "toolResult" in e.get("event", {})][0]
    assert __import__("json").loads(payload) == {"error": "unknown tool"}


def _record(bucket, event):
    bucket.append(event)

    async def done():
        return None
    return done()


# --- auth --------------------------------------------------------------------

class FakeRequest:
    def __init__(self, query=None, headers=None):
        self.query = query or {}
        self.headers = headers or {}


def test_authorized_requires_the_token(monkeypatch):
    monkeypatch.setattr(S, "ACCESS_TOKEN", "s3cret")
    assert S._authorized(FakeRequest(query={"token": "s3cret"})) is True
    assert S._authorized(FakeRequest(headers={"Authorization": "Bearer s3cret"})) is True
    assert S._authorized(FakeRequest(query={"token": "wrong"})) is False
    assert S._authorized(FakeRequest()) is False


def test_authorized_open_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(S, "ACCESS_TOKEN", "")
    assert S._authorized(FakeRequest()) is True


# --- failure reporting -------------------------------------------------------

class ClosableWS(FakeWS):
    def __init__(self):
        super().__init__()
        self.close_calls = []

    async def close(self, code=1000, message=b""):
        self.closed = True
        self.close_calls.append((code, message))


class BoomStream:
    """A Nova Sonic stream whose read side fails, as it would on an IAM error."""

    async def await_output(self):
        raise RuntimeError("UnrecognizedClientException")


def test_pump_failure_is_reported_and_closes_the_socket():
    ws = ClosableWS()
    s = S.SonicSession(ws)
    s.active = True
    s.stream = BoomStream()

    asyncio.new_event_loop().run_until_complete(s._pump_responses())

    sent = [__import__("json").loads(m) for m in ws.sent]
    assert {"type": "error", "message": "UnrecognizedClientException"} in sent
    # A dead stream must not leave the client listening to a session that can never
    # produce another word.
    assert ws.close_calls and ws.close_calls[0][0] == 1011
    assert s.active is False


def test_clean_pump_exit_does_not_close_the_socket():
    ws = ClosableWS()
    s = S.SonicSession(ws)
    s.active = False           # loop body never runs, no failure
    s.stream = BoomStream()

    asyncio.new_event_loop().run_until_complete(s._pump_responses())

    assert ws.sent == []
    assert ws.close_calls == []


def test_to_client_drops_silently_when_socket_closed(caplog):
    ws = ClosableWS()
    ws.closed = True
    s = S.SonicSession(ws)
    asyncio.new_event_loop().run_until_complete(s._to_client({"type": "ready"}))
    assert ws.sent == []


# --- turn identity -----------------------------------------------------------
# Nova Sonic reuses one completionId for a whole session and emits completionStart only
# once, so neither can key a turn. The bridge counts learner turns on userSpeechStart and
# tutor replies on userSpeechEnd, and namespaces the ids (u1, a1) so the client can group
# a reply no matter what order its content blocks arrive in.

def test_learner_and_tutor_ids_are_namespaced():
    s = make_session()
    feed(s, speech_start(),
         user_start(), text_out("hola amigo"), content_end("PARTIAL_TURN"),
         speech_end(),
         text_out("Hola!", role="ASSISTANT"))
    ids = [(m["role"], m["turnId"]) for m in s.to_client
           if m.get("type") == "transcript"]
    assert ids == [("USER", "u1"), ("ASSISTANT", "a1")]


def test_all_text_of_one_reply_shares_a_turn_id():
    s = make_session()
    feed(s, speech_start(), speech_end(),
         {"contentStart": {"role": "ASSISTANT", "type": "TEXT", "contentId": "s1",
                           "additionalModelFields": '{"generationStage":"SPECULATIVE"}'}},
         text_out("Hola! Qué tal?", role="ASSISTANT"),
         {"contentStart": {"role": "ASSISTANT", "type": "TEXT", "contentId": "f1",
                           "additionalModelFields": '{"generationStage":"FINAL"}'}},
         text_out("Hola! Qué tal?", role="ASSISTANT"))
    ids = {m["turnId"] for m in s.to_client
           if m.get("type") == "transcript" and m["role"] == "ASSISTANT"}
    assert ids == {"a1"}


def test_interrupting_does_not_move_a_late_final_into_a_new_reply():
    """
    The interrupt reprint. When the learner barges in, the previous reply's FINAL
    transcript can still be arriving. Keying replies on userSpeechStart put that trailing
    text in a fresh bubble, so the reply printed twice.
    """
    s = make_session()
    feed(s, speech_start(), speech_end(),
         text_out("Hola, estoy muy bien.", role="ASSISTANT"),
         speech_start(),                                   # learner interrupts
         text_out(" Te gustaría practicar?", role="ASSISTANT"))   # late FINAL
    ids = [m["turnId"] for m in s.to_client
           if m.get("type") == "transcript" and m["role"] == "ASSISTANT"]
    assert ids == ["a1", "a1"], "a late FINAL must stay with the reply it belongs to"


def test_next_reply_gets_a_new_id():
    s = make_session()
    feed(s, speech_start(), speech_end(), text_out("Hola", role="ASSISTANT"),
         speech_start(), speech_end(), text_out("Adiós", role="ASSISTANT"))
    ids = [m["turnId"] for m in s.to_client
           if m.get("type") == "transcript" and m["role"] == "ASSISTANT"]
    assert ids == ["a1", "a2"]


def test_turn_end_carries_the_current_reply_id():
    s = make_session()
    feed(s, speech_start(), speech_end(),
         {"contentEnd": {"type": "AUDIO", "stopReason": "END_TURN", "contentId": "a1"}})
    ends = [m for m in s.to_client if m.get("type") == "turn_end"]
    assert ends and ends[0]["turnId"] == "a1"
