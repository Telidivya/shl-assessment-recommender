"""
Lightweight, deterministic conversation understanding.

No external LLM call is used: everything here is rule-based so behavior is
reproducible, auditable, and never invents an assessment that isn't in the
catalog. This trades some flexibility for grounding guarantees, which is
the explicit requirement of the task ("never recommend anything outside
the SHL catalog").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Scope guard
# ---------------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|the|any)?\s*(previous|prior|above)?\s*instructions",
    r"disregard (all|the|any)?\s*(previous|prior|above)?\s*instructions",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions|rules)",
    r"you are now",
    r"forget (that|you are|everything)",
    r"override your",
    r"print your (system|instructions)",
    r"repeat (the|your) (words|text) above",
    r"what (are|were) your (instructions|rules|guidelines)",
]

# Jailbreak framing is distinct from prompt injection: injection tries to
# make the model *execute new instructions embedded in the message*;
# jailbreaking tries to get it to *drop its behavioral constraints entirely*
# (a persona, a "mode", or a claimed absence of rules). Separating them
# means logs/metrics can tell which style of attack is more common, even
# though both currently map to the same refusal category at the API level.
JAILBREAK_PATTERNS = [
    r"\bDAN\b", r"do anything now", r"jailbreak", r"developer mode",
    r"unfiltered (ai|mode)", r"no (restrictions|rules|filters|limitations)",
    r"pretend (you|to be)(?! .*(hiring|recruiter))", r"act as (a|an)(?! (recruiter|hiring manager))",
    r"roleplay as", r"hypothetically,? (you|if you)", r"without (any )?(ethical|safety) (guidelines|restrictions)",
    r"you have no (rules|restrictions|filters)",
]

LEGAL_ADVICE_PATTERNS = [
    r"is it legal", r"legal(ly)? (risk|liab|allowed|required)", r"discriminat",
    r"lawsuit", r"sue (us|them|my|the)", r"employment law", r"labor law",
    r"labour law", r"eeoc", r"comply with .*(law|regulation)", r"gdpr",
    r"can i fire", r"terminate .*(employee|without)", r"visa sponsorship",
    r"protected class", r"adverse impact analysis", r"ofccp",
]

GENERAL_HIRING_ADVICE_PATTERNS = [
    r"how (do|should) i (interview|onboard|negotiate|structure my interview)",
    r"what salary should i (pay|offer)", r"how much should i pay",
    r"write (a|an|me a) job (description|posting|ad)\b(?!.*assessment)",
    r"how do i write a resume", r"interview questions to ask",
    r"how to (fire|terminate|discipline) (an employee|someone)",
    r"performance improvement plan",
    r"general (hiring|recruiting) advice",
    r"how (do|can) i (retain|motivate|manage) (my )?(employees|staff|team)",
    r"onboarding (plan|checklist|process)\b(?!.*assessment)",
]

OFF_TOPIC_HINTS = [
    r"\bweather\b", r"\bstock price\b", r"\brecipe\b", r"write me a poem",
    r"\bpolitical\b opinion", r"who will win the election",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


@dataclass
class ScopeCheck:
    in_scope: bool
    reason: str | None = None


def check_scope(latest_user_message: str) -> ScopeCheck:
    text = latest_user_message.strip()
    if not text:
        return ScopeCheck(True)
    if _matches_any(text, JAILBREAK_PATTERNS):
        return ScopeCheck(False, "jailbreak")
    if _matches_any(text, PROMPT_INJECTION_PATTERNS):
        return ScopeCheck(False, "prompt_injection")
    if _matches_any(text, LEGAL_ADVICE_PATTERNS):
        return ScopeCheck(False, "legal_advice")
    if _matches_any(text, GENERAL_HIRING_ADVICE_PATTERNS):
        return ScopeCheck(False, "general_hiring_advice")
    if _matches_any(text, OFF_TOPIC_HINTS):
        return ScopeCheck(False, "off_topic")
    return ScopeCheck(True)


# ---------------------------------------------------------------------------
# Comparison intent
# ---------------------------------------------------------------------------

COMPARISON_PATTERNS = [
    r"difference between\s+(.+?)\s+and\s+(.+?)[\?\.\!]?$",
    r"compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)[\?\.\!]?$",
    r"(.+?)\s+vs\.?\s+(.+?)[\?\.\!]?$",
    r"(.+?)\s+versus\s+(.+?)[\?\.\!]?$",
    r"how (?:is|does)\s+(.+?)\s+different (?:from|than)\s+(.+?)[\?\.\!]?$",
]


def extract_comparison_targets(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    for pattern in COMPARISON_PATTERNS:
        m = re.search(pattern, stripped, re.IGNORECASE)
        if m and len(m.groups()) == 2:
            a, b = m.group(1).strip(" ?.!"), m.group(2).strip(" ?.!")
            if a and b and a.lower() != b.lower():
                return a, b
    return None


# ---------------------------------------------------------------------------
# Refinement signals (add / remove constraints mid-conversation)
# ---------------------------------------------------------------------------

REFINEMENT_MARKERS = [
    "actually", "instead", "also add", "add ", "remove", "no personality",
    "without", "exclude", "drop the", "change", "rather", "on second thought",
    "swap", "replace",
]

REMOVAL_MARKERS = ["remove", "without", "exclude", "drop the", "no more", "not interested in", "skip the"]


def is_refinement(latest_user_message: str) -> bool:
    text = latest_user_message.lower()
    return any(m in text for m in REFINEMENT_MARKERS)


def wants_removal(latest_user_message: str) -> bool:
    text = latest_user_message.lower()
    return any(m in text for m in REMOVAL_MARKERS)


# ---------------------------------------------------------------------------
# Conversation closing
# ---------------------------------------------------------------------------

CLOSING_PATTERNS = [
    r"^thanks?[!. ]*$", r"^thank you[!. ]*$", r"that('?s| is) all",
    r"that('?s| is) it", r"^no,? (that('?s| is) )?(all|it|everything)",
    r"^(great|perfect|awesome|got it),?\s*thanks?", r"^(nope|no) i'?m (good|all set)",
    r"^i'?m (good|all set|done)", r"^goodbye$", r"^bye$",
]


def is_closing(latest_user_message: str) -> bool:
    text = latest_user_message.strip().lower()
    return any(re.search(p, text) for p in CLOSING_PATTERNS)


# ---------------------------------------------------------------------------
# Context sufficiency scoring (do we know enough to recommend?)
# ---------------------------------------------------------------------------

LEVEL_SIGNALS = [
    "entry-level", "entry level", "junior", "mid-level", "mid level",
    "mid-professional", "senior", "lead", "executive", "graduate", "intern",
    "years", "yrs", "experience", "professional individual contributor",
    "front line", "c-suite", "vp ", "vice president",
]

ROLE_OR_SKILL_SIGNAL_HINTS = [
    # if any of these appear (or a token matches a known tech), treat as a
    # concrete skill/role signal
    "developer", "engineer", "manager", "analyst", "sales", "customer service",
    "administrator", "designer", "accountant", "clerk", "technician",
    "representative", "supervisor", "director", "consultant", "specialist",
]


def word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9]+", text))


def has_level_signal(all_user_text: str) -> bool:
    text = all_user_text.lower()
    return any(sig in text for sig in LEVEL_SIGNALS)


def has_role_or_skill_signal(all_user_text: str) -> bool:
    text = all_user_text.lower()
    return any(sig in text for sig in ROLE_OR_SKILL_SIGNAL_HINTS)


def looks_like_job_description(latest_user_message: str) -> bool:
    """A pasted JD is a strong enough signal to skip further clarification."""
    return word_count(latest_user_message) >= 30
