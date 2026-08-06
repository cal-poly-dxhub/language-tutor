"""
Language profiles and prompt rendering.

The tutor is not Spanish-specific. Everything that names a language, an accent feature
or an example phrase lives in `languages/<name>.json`; the files in `prompts/` are
templates with `{{PLACEHOLDER}}` slots. This module joins the two at synth time so the
containers receive finished prompts and never need to know which language they serve.

Adding a language is copying a profile and translating it — no code changes.
"""

import json
import os
import re

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
LANGUAGES_DIR = os.path.join(ROOT, "languages")
PROMPTS_DIR = os.path.join(ROOT, "prompts")

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z_]+)\}\}")

# Keys a profile must define. Anything starting with "_" in the JSON is a comment.
REQUIRED_FIELDS = (
    "code",
    "targetLanguage",
    "nativeLanguage",
    "voiceId",
    "voiceGender",
    "locale",
    "nativeLocale",
    "espeakLanguage",
    "clarifyPhrase",
    "regionalVariation",
    "learnerErrors",
    "pronunciationExample",
    "grammarExample",
)


class ProfileError(Exception):
    """Raised at synth time so a bad profile fails the deploy, not the conversation."""


def available_languages():
    if not os.path.isdir(LANGUAGES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(LANGUAGES_DIR) if f.endswith(".json"))


def load_profile(name: str) -> dict:
    path = os.path.join(LANGUAGES_DIR, f"{name}.json")
    if not os.path.isfile(path):
        raise ProfileError(
            f"no language profile '{name}' in languages/ "
            f"(available: {', '.join(available_languages()) or 'none'})")
    with open(path, encoding="utf-8") as f:
        try:
            profile = json.load(f)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"{path} is not valid JSON: {exc}") from exc

    missing = [k for k in REQUIRED_FIELDS if not profile.get(k)]
    if missing:
        raise ProfileError(f"{path} is missing required field(s): {', '.join(missing)}")
    for field in ("regionalVariation", "learnerErrors"):
        if not isinstance(profile[field], list) or not profile[field]:
            raise ProfileError(f"{path}: {field} must be a non-empty list of strings")
    # Gendered languages conjugate first-person self-reference, so this has to agree
    # with the chosen voice or the tutor misgenders itself every time it says "I am…".
    if profile["voiceGender"] not in ("female", "male"):
        raise ProfileError(
            f"{path}: voiceGender must be \"female\" or \"male\", "
            f"got {profile['voiceGender']!r}")
    return profile


def substitutions(profile: dict) -> dict:
    """Placeholder name -> replacement text."""
    bullets = "\n".join(f"- {item}" for item in profile["regionalVariation"])
    errors = "\n".join(f"- {item}" for item in profile["learnerErrors"])
    return {
        "TARGET_LANGUAGE": profile["targetLanguage"],
        "NATIVE_LANGUAGE": profile["nativeLanguage"],
        "NATIVE_LANGUAGE_UPPER": profile["nativeLanguage"].upper(),
        "REGIONAL_VARIATION": bullets,
        "LEARNER_ERRORS": errors,
        "CLARIFY_PHRASE": profile["clarifyPhrase"],
        "VOICE_GENDER": profile["voiceGender"],
        "PRONUNCIATION_EXAMPLE": profile["pronunciationExample"],
        "GRAMMAR_EXAMPLE": profile["grammarExample"],
    }


def render(template: str, profile: dict, source: str = "<template>") -> str:
    """
    Fill a prompt template from a profile.

    Unknown or unfilled placeholders raise: a prompt shipped with a literal
    `{{TARGET_LANGUAGE}}` in it would quietly degrade the tutor, so it should break the
    deploy instead.
    """
    values = substitutions(profile)
    unknown = sorted(set(_PLACEHOLDER_RE.findall(template)) - set(values))
    if unknown:
        raise ProfileError(
            f"{source} uses placeholder(s) no profile provides: {', '.join(unknown)}")
    rendered = _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)
    leftover = _PLACEHOLDER_RE.findall(rendered)
    if leftover:
        raise ProfileError(f"{source} still contains placeholders: {leftover}")
    return rendered


def render_prompt(filename: str, profile: dict) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return render(f.read(), profile, source=f"prompts/{filename}")
