"""
Structured representation of what we know about a hiring need, derived from
the *entire* reconstructed conversation on every stateless call.

Why this exists (Priority 3): previously, "do we have enough context?" and
"what should we search for?" were both answered by scattered boolean checks
(`has_role_or_skill_signal`, `has_level_signal`, ad-hoc `detect_signaled_types`
calls) sprinkled through `dialogue.py`. That worked, but every new signal
meant touching orchestration logic again, and there was no single object a
test could assert against.

`HiringRequirements` is that single object. One extractor pass over the
conversation produces it; everything downstream (semantic query
construction, business-rule ranking, and clarification) reads from it
instead of re-deriving signals independently. This is a Single
Responsibility split: extraction is the only place conversation text is
parsed for facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Curated vocabulary rather than "anything the user typed" — keeps skill
# extraction precise and avoids pulling in stopwords/role nouns as "skills".
# Sourced from the catalog's own product names so every extractable skill is
# guaranteed to be findable in the catalog (no skill we'd search for can be
# unmatchable by construction).
KNOWN_SKILL_VOCABULARY = [
    "java", "python", ".net", "c#", "javascript", "angular", "angularjs",
    "android", "aws", "sql", "spring", "node.js", "nodejs", "git", "hadoop",
    "spark", "kafka", "hive", "pig", "hbase", "excel", "photoshop", "swing",
    "agile", "scrum",
]

SENIORITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bentry[- ]level\b", "entry-level"),
    (r"\bintern(ship)?\b", "intern"),
    (r"\bjunior\b", "junior"),
    (r"\bgraduate\b", "graduate"),
    (r"\bmid[- ]level\b|\bmid[- ]professional\b", "mid-professional"),
    (r"\bsenior\b", "senior"),
    (r"\blead\b", "lead"),
    (r"\bmanager\b", "manager"),
    (r"\bdirector\b", "director"),
    (r"\bexecutive\b|\bc-suite\b|\bvp\b", "executive"),
]

ROLE_PATTERN = re.compile(
    r"\b([a-zA-Z][a-zA-Z\-]*(?:\s+[a-zA-Z][a-zA-Z\-]*){0,2}\s+"
    r"(?:developer|engineer|manager|analyst|representative|supervisor|"
    r"director|administrator|designer|accountant|clerk|technician|"
    r"consultant|specialist))\b",
    re.IGNORECASE,
)

PERSONALITY_HINTS = ["personality", "behavioural", "behavioral", "culture fit",
                      "stakeholder", "communication style", "interpersonal", "soft skill"]
COGNITIVE_HINTS = ["cognitive", "reasoning", "aptitude", "numerical", "verbal",
                    "inductive", "deductive", "problem solving", "problem-solving"]
LEADERSHIP_PATTERNS = [
    r"\bleadership\b", r"\blead(ing|s)?\s+(?:\w+\s+){0,2}team\b",
    r"\bmanage\s+(?:\w+\s+){0,2}team\b", r"\bmanage\s+others\b",
    r"\bpeople management\b", r"\bteam lead(er)?\b", r"\bmentor(ing)?\b",
    r"\bdirect reports\b", r"\bsupervis(e|ing|ory)\b",
]
CODING_HINTS = ["coding", "programming", "developer", "hands-on technical",
                "write code", "software engineering"]


ROLE_LEADING_FILLER = re.compile(
    r"^(hiring|hire|need|needs|needed|want|wants|looking for|seeking|for|an?|the)\s+",
    re.IGNORECASE,
)


@dataclass
class HiringRequirements:
    """Everything we've established about the role, derived fresh each call."""

    role: str | None = None
    seniority: str | None = None
    skills: list[str] = field(default_factory=list)
    personality_required: bool = False
    cognitive_required: bool = False
    leadership_required: bool = False
    coding_required: bool = False

    def is_effectively_empty(self) -> bool:
        return not (
            self.role or self.seniority or self.skills
            or self.personality_required or self.cognitive_required
            or self.leadership_required or self.coding_required
        )

    def as_query_text(self) -> str:
        """Render as a normalized natural-language string for embedding.

        Structured -> text rather than raw transcript -> text, so retrieval
        is driven by *what we've confirmed*, not by incidental phrasing or
        stray words that happened to appear in the conversation.
        """
        parts: list[str] = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        if self.skills:
            parts.append("skills: " + ", ".join(self.skills))
        focus = []
        if self.personality_required:
            focus.append("personality and behavioral fit")
        if self.cognitive_required:
            focus.append("cognitive ability and reasoning")
        if self.leadership_required:
            focus.append("leadership and people management")
        if self.coding_required:
            focus.append("coding and technical skill")
        if focus:
            parts.append("assessment focus: " + ", ".join(focus))
        return ". ".join(parts) if parts else ""


class RequirementsExtractor:
    """Parses the full user-turn transcript into a `HiringRequirements`."""

    def extract(self, user_messages: list[str]) -> HiringRequirements:
        full_text = " ".join(user_messages)
        text_lower = full_text.lower()

        return HiringRequirements(
            role=self._extract_role(full_text),
            seniority=self._extract_seniority(text_lower),
            skills=self._extract_skills(text_lower),
            personality_required=self._any_hint(text_lower, PERSONALITY_HINTS),
            cognitive_required=self._any_hint(text_lower, COGNITIVE_HINTS),
            leadership_required=self._any_pattern(text_lower, LEADERSHIP_PATTERNS),
            coding_required=self._any_hint(text_lower, CODING_HINTS),
        )

    @staticmethod
    def _extract_role(text: str) -> str | None:
        match = ROLE_PATTERN.search(text)
        if not match:
            return None
        role = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        # Strip lead-in filler repeatedly: "hiring a java developer" first
        # loses "hiring ", then "a ", leaving the clean "java developer".
        # A single pass isn't enough since fillers can stack (verb + article).
        while True:
            stripped = ROLE_LEADING_FILLER.sub("", role)
            if stripped == role:
                break
            role = stripped
        return role or None

    @staticmethod
    def _extract_seniority(text_lower: str) -> str | None:
        for pattern, label in SENIORITY_PATTERNS:
            if re.search(pattern, text_lower):
                return label
        return None

    @staticmethod
    def _extract_skills(text_lower: str) -> list[str]:
        found = []
        for skill in KNOWN_SKILL_VOCABULARY:
            # word-boundary-safe check that still works for tokens with dots
            # (e.g. ".net", "node.js") where \b doesn't behave as expected.
            if skill in text_lower:
                found.append(skill)
        return found

    @staticmethod
    def _any_hint(text_lower: str, hints: list[str]) -> bool:
        return any(h in text_lower for h in hints)

    @staticmethod
    def _any_pattern(text_lower: str, patterns: list[str]) -> bool:
        return any(re.search(p, text_lower) for p in patterns)
