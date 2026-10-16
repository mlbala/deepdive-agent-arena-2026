"""Deterministic detectors for prompt injection (direct + indirect) and harmful requests.

Design notes
------------
* Text is NFKC-normalised and stripped of zero-width/format characters before matching
  (see `document_ingestor.clean_text`), which defeats the common homoglyph / invisible
  character obfuscations.
* Harmful-request patterns target *operational* intent (synthesise, build, step-by-step)
  so that legitimate research about the same topics (treaties, history, detection,
  prevention, policy) is not blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_F = re.IGNORECASE | re.DOTALL


@dataclass(frozen=True)
class PatternHit:
    category: str
    name: str
    start: int
    end: int


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, _F)


# --- Prompt injection (user input and retrieved content) ------------------------
INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": _rx(
        r"\b(ignore|disregard|forget|override|bypass)\b\W+(?:\w+\W+){0,2}?"
        r"(previous|prior|above|earlier|preceding|your|system|developer|safety)\W+(?:\w+\W+){0,1}?"
        r"(instructions?|prompts?|rules|directives|guidelines|guardrails|constraints|messages?)\b"
    ),
    "role_hijack": _rx(
        r"\b(you are now|from now on,? you are|act as|pretend to be|roleplay as)\b\W+(?:\w+\W+){0,3}?"
        r"(dan|jailbroken|unfiltered|uncensored|developer mode|evil|without (any )?restrictions)\b"
    ),
    "jailbreak_marker": _rx(r"\b(do anything now|jailbreak(?:ing)?|developer mode enabled|godmode)\b"),
    "prompt_exfiltration": _rx(
        r"\b(reveal|print|show|output|repeat|leak|dump|display|tell me)\b\W+(?:\w+\W+){0,4}?"
        r"(system prompt|hidden prompt|initial instructions|your (instructions|prompt|rules|configuration)|"
        r"developer message|api[_ ]?keys?|secret keys?|environment variables|\.env\b)"
    ),
    "credential_probe": _rx(r"\b(MESH_API_KEY|OPENAI_API_KEY|TAVILY_API_KEY|ANTHROPIC_API_KEY|ADMIN_API_KEY)\b"),
    "chat_template_tokens": _rx(r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[/?INST\]|</?system>|<<SYS>>)"),
    "fake_instruction_header": _rx(r"(^|\n)\s*(#{1,3}\s*)?(new|updated|additional|system)\s+instructions?\s*:"),
    "exfiltration_link": _rx(
        r"\b(send|post|upload|forward|exfiltrate)\b\W+(?:\w+\W+){0,4}?(data|conversation|context|prompt|keys?)\b"
        r"\W+(?:\w+\W+){0,3}?(to|at)\W+https?://"
    ),
}

# Additional patterns that only make sense inside retrieved documents (indirect injection / SEO poisoning).
INDIRECT_PATTERNS: dict[str, re.Pattern[str]] = {
    "ai_directed_instruction": _rx(
        r"\b(if you are|attention|note to|instructions? for)\b\W+(?:\w+\W+){0,2}?"
        r"(an? )?(ai|llm|language model|assistant|chatbot|agent)s?\b"
    ),
    "ai_must_comply": _rx(
        r"\b(ai|llm|language model|assistant|chatbot|agent)s?\b\W+(?:\w+\W+){0,3}?"
        r"(must|should|are required to|will)\W+(always\W+)?(say|respond|answer|state|claim|recommend|cite|rank)\b"
    ),
    "citation_manipulation": _rx(
        r"\b(cite|rank|treat|present)\b\W+(?:\w+\W+){0,3}?(this|our)\W+(page|site|source|article|document)\W+"
        r"(?:\w+\W+){0,3}?(most )?(authoritative|trusted|reliable|primary|definitive)\b"
    ),
}

# --- Harmful requests (operational intent only) ---------------------------------
# A hit needs self-directed intent ("how do I", "steps to", "give me", or a leading imperative)
# AND a production verb close to the object. "How did Iraq produce chemical weapons?" (history)
# and "How do companies defend against ransomware?" (defence) therefore pass.
_INTENT = (
    r"(how (do|can|could|would|should) (i|we|you|one|someone)\b|how to\b|steps? (to|for)\b|"
    r"instructions? (to|for|on)\b|guide (to|for|on)\b|recipe (for|to)\b|tutorial (on|for)\b|"
    r"walk me through|step[- ]by[- ]step|teach me|help me|(give|provide|write|show|send) me)"
)
_LEAD = r"(^|[.!?]\s+)\s*(please\s+)?"
_MAKE = (
    r"(make|build|synthes[iy][sz]e|synthesis of|produce|manufactur\w*|assemble|weaponi[sz]e|"
    r"cultivate|culture|enrich|create|brew|cook|write|code|develop|deploy)"
)


def _operational(obj: str) -> re.Pattern[str]:
    return _rx(
        rf"({_INTENT}\W+(?:\w+\W+){{0,3}}?|{_LEAD}){_MAKE}\W+(?:\w+\W+){{0,4}}?{obj}"
    )


HARM_PATTERNS: dict[str, re.Pattern[str]] = {
    "weapons_cbrn": _operational(
        r"(sarin|soman|tabun|vx\b|novichok|nerve agents?|mustard gas|ricin|anthrax|botulinum|smallpox|"
        r"bio-?weapons?|chemical weapons?|dirty bombs?|radiological weapons?|nuclear (bomb|weapon|device)s?|"
        r"weapons?-grade (uranium|plutonium))"
    ),
    "explosives": _operational(
        r"(pipe bombs?|explosive devices?|\bied\b|car bombs?|nail bombs?|tatp\b|detonators?|"
        r"bombs?\b(?!\s+(shelters?|squads?|disposal|threats?|cyclones?|calorimeters?)))"
    ),
    "malware": _operational(
        r"(ransomware|keyloggers?|botnets?|credential stealers?|info-?stealers?|rootkits?|"
        r"self-propagating (worm|virus)|zero-click exploits?)"
    ),
    "self_harm": _rx(
        r"(how (do|can|should) i|best way to|easiest way to|painless way to|methods? (of|to))\W+(?:\w+\W+){0,2}?"
        r"(kill myself|commit suicide|end my life|take my own life|self[- ]harm|hurt myself)"
    ),
    "child_sexual_content": _rx(
        r"\b(generate|write|create|draw|show|describe|produce|send|find|share)\b\W+(?:\w+\W+){0,5}?"
        r"((sexual|nude|naked|porn\w*|explicit|erotic)\W+(?:\w+\W+){0,4}?"
        r"(child|children|minors?|underage|kids?|pre-?teens?)\b|"
        r"(child|children|minors?|underage|kids?|pre-?teens?)\W+(?:\w+\W+){0,4}?"
        r"(nude|naked|porn\w*|explicit|erotic|sexual (content|images?|stories|material)))"
    ),
    "violent_extremism": _rx(
        rf"{_INTENT}\W+(?:\w+\W+){{0,4}}?(plan|organi[sz]e|carry out|execute|commit)\W+(?:\w+\W+){{0,3}}?"
        r"(terrorist attacks?|mass shootings?|massacres?|bombings?)"
    ),
}

HARM_MESSAGES = {
    "self_harm": (
        "This request looks like it may involve self-harm. DeepDive will not research methods of self-harm. "
        "If you are struggling, please reach out to a local emergency number or a crisis line "
        "(for example 988 in the US, or iCall +91 9152987821 in India)."
    ),
}


def _scan(text: str, patterns: dict[str, re.Pattern[str]], category: str) -> list[PatternHit]:
    return [
        PatternHit(category, name, m.start(), m.end())
        for name, pattern in patterns.items()
        for m in pattern.finditer(text)
    ]


def detect_injection(text: str, *, indirect: bool = False) -> list[PatternHit]:
    hits = _scan(text, INJECTION_PATTERNS, "prompt_injection")
    if indirect:
        hits += _scan(text, INDIRECT_PATTERNS, "prompt_injection")
    return sorted(hits, key=lambda h: h.start)


def detect_harm(text: str) -> list[PatternHit]:
    return sorted(_scan(text, HARM_PATTERNS, "harmful_request"), key=lambda h: h.start)
