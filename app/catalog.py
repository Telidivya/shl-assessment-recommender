"""
Loads the SHL Individual Test Solutions catalog and provides:

1. Structured `Product` records (now including optional job-level and
   duration metadata, used by the ranking and comparison layers).
2. A deterministic keyword search (`Catalog.search`), retained as a
   dependency-free fallback path and as the source of `detect_signaled_types`,
   which the ranker uses to build business-rule boost/exclude sets.

The primary recommendation path is now `app/retrieval.py`'s
`SemanticRetriever` + `app/ranking.py`'s `RequirementsAwareRanker`, both of
which take a `Catalog` (or its `.products`) as input. This module stays
focused on one job — loading and indexing catalog *data* — per the Single
Responsibility Principle; it has no opinion on how retrieval or ranking work.

The catalog is loaded once at process startup from data/shl_catalog.json.
Every recommendation the agent makes is drawn from this in-memory list,
so it is structurally impossible for the agent to surface a URL that
didn't come from the scraped catalog.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .exceptions import CatalogLoadError

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "shl_catalog.json"

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "is", "are", "be", "we", "i", "am", "im", "need", "needs", "needed",
    "want", "wants", "looking", "hire", "hiring", "role", "job", "position",
    "candidate", "candidates", "test", "tests", "assessment", "assessments",
    "please", "some", "any", "that", "this", "who", "what", "how", "about",
    "level", "years", "year", "experience", "someone", "person", "our",
    "will", "work", "works", "working", "also", "add", "actually", "just",
    # Generic role nouns: useful for the sufficiency heuristic in nlu.py
    # (checked against raw text there) but too generic/noisy for keyword
    # overlap matching here — e.g. "manager" alone would otherwise match
    # unrelated products like "Adobe Experience Manager".
    "developer", "engineer", "manager", "analyst", "representative",
    "supervisor", "director", "administrator", "designer", "accountant",
    "clerk", "technician", "consultant", "specialist",
}

TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# Maps free-text signals a user might use to the catalog's test_type codes.
TEST_TYPE_SIGNALS = {
    "P": ["personality", "behavioural", "behavioral", "culture fit", "style",
          "opq", "motivation", "motivational", "soft skill", "soft skills",
          "stakeholder", "communication style", "interpersonal", "temperament"],
    "A": ["cognitive", "aptitude", "reasoning", "verify", "numerical", "verbal",
          "inductive", "deductive", "logical", "iq", "ability", "gsa" if False else "g+",
          "general ability", "problem solving", "problem-solving"],
    "K": ["coding", "programming", "software engineering", "technical skills",
          "knowledge test", "language test", "sql", "java", "python", ".net",
          "javascript", "aws", "cloud computing", "framework"],
    "S": ["simulation", "hands-on", "hands on", "realistic task", "in-basket",
          "in basket", "practical exercise"],
    "B": ["situational judgement", "situational judgment", "sjt", "biodata",
          "global skills", "gsa", "workplace behavior", "workplace behaviour"],
    "C": ["competency", "competencies", "competence"],
    "D": ["360", "development", "multi-rater", "feedback", "leadership development"],
    "E": ["assessment center", "assessment centre", "development center",
          "development centre", "exercise", "role play", "role-play"],
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


_DURATION_PATTERN = re.compile(r"(\d{1,3})\s*[- ]?\s*minutes?\b", re.IGNORECASE)


@dataclass
class Product:
    name: str
    url: str
    test_type: list[str]
    description: str = ""
    job_levels: list[str] = field(default_factory=list)
    duration_minutes: int | None = None
    _tokens: set[str] = field(default_factory=set, repr=False, compare=False)

    def __post_init__(self) -> None:
        blob = f"{self.name} {self.description}"
        self._tokens = set(_tokenize(blob))
        if self.duration_minutes is None:
            # Derive from the product's own (already human-verified)
            # description rather than asking a maintainer to duplicate the
            # number in a separate field — one source of truth, and no new
            # fact is introduced that wasn't already in verified catalog text.
            match = _DURATION_PATTERN.search(self.description)
            if match:
                self.duration_minutes = int(match.group(1))

    def type_labels(self) -> list[str]:
        return [TEST_TYPE_LEGEND.get(t, t) for t in self.test_type]

    def as_recommendation(self) -> dict[str, str]:
        """Serializes to exactly the API's Recommendation schema.

        Deliberately returns only the three fields the API contract
        promises (name/url/test_type) even though `Product` itself now
        carries more structured data — job_levels/duration power ranking
        and comparison, but the public schema is unchanged per spec.
        """
        return {
            "name": self.name,
            "url": self.url,
            "test_type": "".join(self.test_type),
        }


class Catalog:
    """In-memory, read-only view of the SHL Individual Test Solutions catalog."""

    def __init__(self, path: Path = DATA_PATH):
        self.products: list[Product] = self._load(path)
        self._by_name_lower: dict[str, Product] = {p.name.lower(): p for p in self.products}
        logger.info("Catalog loaded: %d products from %s", len(self.products), path)

    @staticmethod
    def _load(path: Path) -> list[Product]:
        """Reads and validates the catalog file.

        Raises `CatalogLoadError` (rather than letting a `KeyError` /
        `json.JSONDecodeError` / `FileNotFoundError` propagate) for a
        deliberate reason: `main.py`'s startup hook treats this exception
        as fatal-and-specific, and it's the exact scenario Priority 8's
        "invalid catalog" test exercises. A vague built-in exception would
        work too, but wouldn't distinguish "catalog problem" from any other
        startup failure in logs or in tests.
        """
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError as exc:
            raise CatalogLoadError(f"Catalog file not found at {path}") from exc
        except json.JSONDecodeError as exc:
            raise CatalogLoadError(f"Catalog file at {path} is not valid JSON: {exc}") from exc

        if not isinstance(raw, dict) or "products" not in raw:
            raise CatalogLoadError(f"Catalog file at {path} is missing a top-level 'products' list")

        products: list[Product] = []
        for i, entry in enumerate(raw["products"]):
            try:
                products.append(
                    Product(
                        name=entry["name"],
                        url=entry["url"],
                        test_type=list(entry.get("test_type", [])),
                        description=entry.get("description", ""),
                        job_levels=list(entry.get("job_levels", [])),
                        duration_minutes=entry.get("duration_minutes"),
                    )
                )
            except KeyError as exc:
                raise CatalogLoadError(
                    f"Catalog entry #{i} at {path} is missing required field {exc}"
                ) from exc

        if not products:
            raise CatalogLoadError(f"Catalog file at {path} contains zero products")

        return products

    def __len__(self) -> int:
        return len(self.products)

    def find_by_name_fragment(self, fragment: str) -> Product | None:
        """Fuzzy-ish lookup: exact, then substring, then acronym match."""
        frag = fragment.strip().lower()
        if not frag:
            return None
        if frag in self._by_name_lower:
            return self._by_name_lower[frag]
        # substring match either direction
        best = None
        for p in self.products:
            name_l = p.name.lower()
            if frag in name_l or name_l in frag:
                if best is None or len(name_l) < len(best.name):
                    best = p
        if best:
            return best
        # acronym match, e.g. "opq" -> "Occupational Personality Questionnaire OPQ32r"
        for p in self.products:
            initials = "".join(w[0] for w in re.findall(r"[a-zA-Z0-9]+", p.name)).lower()
            if frag in initials or frag in p.name.lower().replace(" ", ""):
                return p
        return None

    def score(
        self,
        product: Product,
        query_tokens: Iterable[str],
        boosted_types: set[str],
        excluded_types: set[str],
    ) -> float:
        """Deterministic fallback scorer (used when semantic retrieval is
        unavailable, and by tests that want a dependency-free code path)."""
        if excluded_types and set(product.test_type) & excluded_types:
            return -1.0
        qtokens = set(query_tokens)
        overlap = len(qtokens & product._tokens)
        score = float(overlap)
        if boosted_types and set(product.test_type) & boosted_types:
            score += 1.0
        # small boost for exact name-word hits (e.g. "java" hitting "Java 8")
        for t in qtokens:
            if len(t) >= 3 and t in product.name.lower():
                score += 1.5
        return score

    def search(
        self,
        query: str,
        boosted_types: set[str] | None = None,
        excluded_types: set[str] | None = None,
        top_k: int = 5,
    ) -> list[Product]:
        """Keyword-overlap search. Retained as the fallback retrieval path;
        see `app.retrieval.SemanticRetriever` for the primary path."""
        boosted_types = boosted_types or set()
        excluded_types = excluded_types or set()
        qtokens = _tokenize(query)
        scored = [
            (self.score(p, qtokens, boosted_types, excluded_types), p)
            for p in self.products
        ]
        scored = [(s, p) for s, p in scored if s > 0]
        scored.sort(key=lambda sp: sp[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def detect_signaled_types(self, text: str) -> set[str]:
        """Maps free-text type language (e.g. "personality") to test_type codes."""
        text_l = text.lower()
        hits: set[str] = set()
        for code, signals in TEST_TYPE_SIGNALS.items():
            for sig in signals:
                if sig and sig in text_l:
                    hits.add(code)
                    break
        return hits


_catalog_singleton: Catalog | None = None


def get_catalog() -> Catalog:
    """Process-wide catalog singleton.

    A module-level singleton (rather than re-reading the JSON file per
    request) is a deliberate performance choice (Priority 9): the catalog
    is small and immutable for the lifetime of the process, so there is no
    correctness reason to reload it, and every request avoids redundant
    disk + JSON-parse + tokenization work.
    """
    global _catalog_singleton
    if _catalog_singleton is None:
        _catalog_singleton = Catalog()
    return _catalog_singleton
