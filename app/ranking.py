"""
Multi-factor ranking on top of semantic retrieval.

Priority 2 asks for semantic similarity + skill overlap + job level +
assessment type + business rules, combined in an *explainable* way. Pure
embedding similarity is a good recall mechanism but a poor final ranking
signal on its own — two products can be semantically close to a query for
the wrong reason (e.g. both mention "team" in unrelated contexts). This
module is the precision layer: it takes the Top-20 semantic candidates and
re-scores them against what we've actually confirmed about the role.

Every score is returned as a `RankedCandidate` with a `breakdown` dict, so
"why was this recommended" is always answerable from the object itself,
even though the public API schema (unchanged, per the constraint) only
ever surfaces name/url/test_type. The breakdown is for logs, debugging, and
future observability — not a schema change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .catalog import Product
from .requirements import HiringRequirements

logger = logging.getLogger(__name__)

# Weights are intentionally simple (sum to 1.0 for the continuous factors)
# rather than learned — with no labeled relevance data to fit against, a
# hand-set linear combination is the honest choice: easy to reason about,
# easy to tune from observed failures, and each term is independently
# inspectable in the breakdown.
WEIGHT_SEMANTIC = 0.45
WEIGHT_SKILL_OVERLAP = 0.25
WEIGHT_LEVEL_MATCH = 0.10
WEIGHT_TYPE_MATCH = 0.20

BUSINESS_RULE_EXCLUDE_PENALTY = -1_000.0  # effectively removes excluded types
BUSINESS_RULE_BOOST = 0.15


@dataclass
class RankedCandidate:
    product: Product
    final_score: float
    breakdown: dict[str, float] = field(default_factory=dict)

    def explanation(self) -> str:
        parts = [f"{k}={v:.2f}" for k, v in self.breakdown.items()]
        return f"{self.product.name}: " + ", ".join(parts)


class RequirementsAwareRanker:
    """Re-ranks semantic candidates using structured hiring requirements."""

    def rank(
        self,
        candidates: list[tuple[Product, float]],
        requirements: HiringRequirements,
        boosted_types: set[str],
        excluded_types: set[str],
        top_k: int,
    ) -> list[RankedCandidate]:
        ranked = [
            self._score_one(product, semantic_score, requirements, boosted_types, excluded_types)
            for product, semantic_score in candidates
        ]
        ranked = [c for c in ranked if c.final_score > BUSINESS_RULE_EXCLUDE_PENALTY / 2]
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked[:top_k]

    def _score_one(
        self,
        product: Product,
        semantic_score: float,
        requirements: HiringRequirements,
        boosted_types: set[str],
        excluded_types: set[str],
    ) -> RankedCandidate:
        skill_overlap = self._skill_overlap_score(product, requirements)
        level_match = self._level_match_score(product, requirements)
        type_match = self._type_match_score(product, requirements, boosted_types)
        business_rule = self._business_rule_adjustment(product, excluded_types, boosted_types)

        final = (
            WEIGHT_SEMANTIC * semantic_score
            + WEIGHT_SKILL_OVERLAP * skill_overlap
            + WEIGHT_LEVEL_MATCH * level_match
            + WEIGHT_TYPE_MATCH * type_match
            + business_rule
        )
        breakdown = {
            "semantic_similarity": semantic_score,
            "skill_overlap": skill_overlap,
            "level_match": level_match,
            "type_match": type_match,
            "business_rule": business_rule,
            "final_score": final,
        }
        return RankedCandidate(product=product, final_score=final, breakdown=breakdown)

    @staticmethod
    def _skill_overlap_score(product: Product, requirements: HiringRequirements) -> float:
        if not requirements.skills:
            return 0.0
        haystack = f"{product.name} {product.description}".lower()
        hits = sum(1 for skill in requirements.skills if skill in haystack)
        return hits / len(requirements.skills)

    @staticmethod
    def _level_match_score(product: Product, requirements: HiringRequirements) -> float:
        # The seed catalog doesn't have structured job-level metadata for
        # every item (see Product.job_levels); when we don't know, we stay
        # neutral (0.5) rather than penalizing — an unknown level should
        # never be scored the same as a *confirmed* mismatch.
        if not requirements.seniority or not product.job_levels:
            return 0.5
        seniority = requirements.seniority.lower()
        levels = [lvl.lower() for lvl in product.job_levels]
        if any(seniority in lvl or lvl in seniority for lvl in levels):
            return 1.0
        return 0.2

    @staticmethod
    def _type_match_score(
        product: Product, requirements: HiringRequirements, boosted_types: set[str]
    ) -> float:
        wanted_types: set[str] = set(boosted_types)
        if requirements.personality_required:
            wanted_types.add("P")
        if requirements.cognitive_required:
            wanted_types.add("A")
        if requirements.coding_required:
            wanted_types.add("K")
        if requirements.leadership_required:
            wanted_types.update({"D", "C"})
        if not wanted_types:
            return 0.5  # no explicit type preference stated — stay neutral
        return 1.0 if set(product.test_type) & wanted_types else 0.0

    @staticmethod
    def _business_rule_adjustment(
        product: Product, excluded_types: set[str], boosted_types: set[str]
    ) -> float:
        if excluded_types and set(product.test_type) & excluded_types:
            return BUSINESS_RULE_EXCLUDE_PENALTY
        if boosted_types and set(product.test_type) & boosted_types:
            return BUSINESS_RULE_BOOST
        return 0.0
