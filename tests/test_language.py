"""
Tests for language profiles and prompt rendering.

The contract: no language is baked into code or prompts. Every profile in `languages/`
renders every prompt with no placeholder left behind, and a broken profile fails at
synth time rather than degrading the tutor at runtime.
"""

import json
import os
import pathlib
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from cdk import language as L  # noqa: E402

PROMPTS = ("conversation.txt", "coach.txt")

# Every voice id Nova 2 Sonic accepts, from the language-support page. A typo here is
# rejected by the model at session start, which is an awful place to find out.
VALID_VOICES = {
    "tiffany", "matthew",              # en-US, and polyglot across all languages
    "amy",                             # en-GB
    "olivia",                          # en-AU
    "kiara", "arjun",                  # en-IN and hi-IN
    "ambre", "florian",                # fr-FR
    "beatrice", "lorenzo",             # it-IT
    "tina", "lennart",                 # de-DE
    "lupe", "carlos",                  # es-US
    "carolina", "leo",                 # pt-BR
}

# The languages Nova 2 Sonic speaks. A profile should exist for each, so the tutor can
# teach anything the model supports.
SUPPORTED_LANGUAGES = {
    "english", "french", "italian", "german", "spanish", "portuguese", "hindi",
}


def test_shipped_languages_are_discovered():
    found = L.available_languages()
    assert "spanish" in found
    # A second profile exists to keep the templates honest.
    assert len(found) >= 2


def test_every_language_nova_sonic_speaks_has_a_profile():
    missing = SUPPORTED_LANGUAGES - set(L.available_languages())
    assert not missing, f"no profile for: {', '.join(sorted(missing))}"


@pytest.mark.parametrize("name", L.available_languages())
def test_voice_id_is_one_nova_sonic_accepts(name):
    voice = L.load_profile(name)["voiceId"]
    assert voice in VALID_VOICES, f"{name}.json uses unknown voice {voice!r}"


# The feminine/masculine split from the AWS voice table. Gendered languages conjugate
# first-person self-reference, so a mismatch makes the tutor misgender itself.
FEMININE = {"tiffany", "amy", "olivia", "kiara", "ambre", "beatrice", "tina", "lupe",
            "carolina"}
MASCULINE = {"matthew", "arjun", "florian", "lorenzo", "lennart", "carlos", "leo"}


@pytest.mark.parametrize("name", L.available_languages())
def test_voice_gender_matches_the_voice(name):
    profile = L.load_profile(name)
    expected = "female" if profile["voiceId"] in FEMININE else "male"
    assert profile["voiceGender"] == expected, (
        f"{name}.json says {profile['voiceGender']} but voice "
        f"{profile['voiceId']!r} is {expected}")


def test_bad_voice_gender_is_rejected(tmp_path, monkeypatch):
    good = json.loads((pathlib.Path(ROOT) / "languages" / "spanish.json").read_text())
    good["voiceGender"] = "feminine"
    monkeypatch.setattr(L, "LANGUAGES_DIR", str(tmp_path))
    (tmp_path / "broken.json").write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(L.ProfileError) as exc:
        L.load_profile("broken")
    assert "voiceGender" in str(exc.value)


@pytest.mark.parametrize("name", L.available_languages())
def test_conversation_prompt_demands_language_mirroring(name):
    """
    Without an explicit, top-priority mirroring rule, Nova 2 Sonic answers a
    native-language question in the target language — verified against the live model,
    including two weaker phrasings that it ignored. What finally worked was a
    language-neutral persona plus a rule that says it overrides everything below it.
    """
    profile = L.load_profile(name)
    rendered = L.render_prompt("conversation.txt", profile)
    assert "ANSWER IN WHICHEVER LANGUAGE THEY USED" in rendered
    assert "THIS OVERRIDES EVERYTHING BELOW" in rendered
    assert "Never translate their words" in rendered
    assert profile["nativeLanguage"] in rendered
    # The persona must not itself declare a language, or it outweighs the rule above.
    first_line = rendered.splitlines()[0]
    assert profile["targetLanguage"] not in first_line, (
        f"the opening line names {profile['targetLanguage']}, which competes with the "
        f"mirroring rule")


@pytest.mark.parametrize("name", L.available_languages())
def test_mirroring_covers_greetings(name):
    """
    The failure that mattered in practice: a bare "hello" was answered in the target
    language, and once the tutor had spoken it, later native-language utterances came
    back mistranscribed as the target language. Greetings have to mirror too.
    """
    rendered = L.render_prompt("conversation.txt", L.load_profile(name))
    assert "THIS INCLUDES GREETINGS AND SMALL TALK" in rendered
    assert "drift" in rendered


@pytest.mark.parametrize("name", L.available_languages())
def test_conversation_prompt_states_the_assistant_gender(name):
    profile = L.load_profile(name)
    rendered = L.render_prompt("conversation.txt", profile)
    assert profile["voiceGender"] in rendered


@pytest.mark.parametrize("name", L.available_languages())
def test_clarify_phrase_is_not_left_in_english(name):
    profile = L.load_profile(name)
    if profile["targetLanguage"] == profile["nativeLanguage"]:
        return                     # English teaching English: nothing to translate
    english = L.load_profile("english")["clarifyPhrase"]
    assert profile["clarifyPhrase"] != english, (
        f"{name}.json still has the English clarify phrase")


@pytest.mark.parametrize("name", L.available_languages())
def test_every_profile_loads_and_has_required_fields(name):
    profile = L.load_profile(name)
    for field in L.REQUIRED_FIELDS:
        assert profile.get(field), f"{name}.json missing {field}"
    assert isinstance(profile["regionalVariation"], list)
    assert profile["regionalVariation"], f"{name}.json has no regional variation"


@pytest.mark.parametrize("name", L.available_languages())
@pytest.mark.parametrize("prompt", PROMPTS)
def test_every_prompt_renders_for_every_language(name, prompt):
    rendered = L.render_prompt(prompt, L.load_profile(name))
    assert "{{" not in rendered and "}}" not in rendered
    assert L.load_profile(name)["targetLanguage"] in rendered


def test_rendered_prompts_differ_between_languages():
    spanish = L.render_prompt("coach.txt", L.load_profile("spanish"))
    french = L.render_prompt("coach.txt", L.load_profile("french"))
    assert spanish != french
    assert "Spanish" in spanish and "Spanish" not in french
    assert "French" in french


def test_prompt_templates_name_no_language():
    # The templates themselves must stay neutral; if a language name is written into
    # one, switching profiles silently produces a contradictory prompt.
    for prompt in PROMPTS:
        text = open(os.path.join(ROOT, "prompts", prompt), encoding="utf-8").read()
        for token in ("Spanish", "French", "español", "seseo", "yeísmo", "voseo"):
            assert token not in text, f"{prompt} mentions {token!r}"


def test_regional_variation_renders_as_a_bullet_list():
    profile = L.load_profile("spanish")
    rendered = L.render_prompt("coach.txt", profile)
    for item in profile["regionalVariation"]:
        assert f"- {item}" in rendered


def test_unknown_language_is_rejected():
    with pytest.raises(L.ProfileError) as exc:
        L.load_profile("klingon")
    assert "klingon" in str(exc.value)
    assert "spanish" in str(exc.value)      # lists what is available


def test_missing_field_is_rejected(tmp_path, monkeypatch):
    bad = {"code": "xx", "targetLanguage": "Xhosa"}
    monkeypatch.setattr(L, "LANGUAGES_DIR", str(tmp_path))
    (tmp_path / "broken.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(L.ProfileError) as exc:
        L.load_profile("broken")
    assert "missing required field" in str(exc.value)


def test_invalid_json_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LANGUAGES_DIR", str(tmp_path))
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(L.ProfileError) as exc:
        L.load_profile("broken")
    assert "not valid JSON" in str(exc.value)


def test_unknown_placeholder_is_rejected():
    profile = L.load_profile("spanish")
    with pytest.raises(L.ProfileError) as exc:
        L.render("Teach {{TARGET_LANGUAGE}} using {{SECRET_SAUCE}}.", profile)
    assert "SECRET_SAUCE" in str(exc.value)


def test_render_substitutes_all_known_placeholders():
    profile = L.load_profile("spanish")
    template = " ".join("{{%s}}" % key for key in L.substitutions(profile))
    rendered = L.render(template, profile)
    assert "{{" not in rendered
    assert profile["clarifyPhrase"] in rendered
    assert profile["nativeLanguage"].upper() in rendered


@pytest.mark.parametrize("name", L.available_languages())
def test_profiles_list_errors_that_must_be_flagged(name):
    """
    The counterweight to regionalVariation. With only a "never flag" list the reviewer
    excuses almost everything as accent — verified on real speech, a pronounced silent "h"
    produced no note at 50% phoneme accuracy until the profile said it was an error.
    """
    profile = L.load_profile(name)
    assert profile["learnerErrors"], f"{name}.json lists no learner errors"
    rendered = L.render_prompt("coach.txt", profile)
    assert "ALWAYS WORTH FLAGGING" in rendered
    for item in profile["learnerErrors"]:
        assert f"- {item}" in rendered


@pytest.mark.parametrize("name", L.available_languages())
def test_learner_errors_and_regional_variation_are_distinct(name):
    profile = L.load_profile(name)
    assert not set(profile["learnerErrors"]) & set(profile["regionalVariation"]), (
        f"{name}.json lists the same item as both an error and acceptable variation")


def test_empty_learner_errors_is_rejected(tmp_path, monkeypatch):
    good = json.loads((pathlib.Path(ROOT) / "languages" / "spanish.json").read_text())
    good["learnerErrors"] = []
    monkeypatch.setattr(L, "LANGUAGES_DIR", str(tmp_path))
    (tmp_path / "broken.json").write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(L.ProfileError) as exc:
        L.load_profile("broken")
    assert "learnerErrors" in str(exc.value)
