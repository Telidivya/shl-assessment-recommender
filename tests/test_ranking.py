from app.catalog import Product
from app.ranking import RequirementsAwareRanker
from app.requirements import HiringRequirements


def _java_product() -> Product:
    return Product(name="Java 8 (New)", url="https://www.shl.com/x/java/", test_type=["K"],
                   description="Knowledge test on Java programming.")


def _opq_product() -> Product:
    return Product(name="OPQ32r", url="https://www.shl.com/x/opq/", test_type=["P"],
                   description="Personality assessment measuring workplace behavioral style.")


def test_skill_overlap_rewards_matching_skill():
    ranker = RequirementsAwareRanker()
    req = HiringRequirements(skills=["java"])
    ranked = ranker.rank([(_java_product(), 0.5)], req, boosted_types=set(), excluded_types=set(), top_k=5)
    assert ranked[0].breakdown["skill_overlap"] == 1.0


def test_excluded_type_removes_candidate_entirely():
    ranker = RequirementsAwareRanker()
    req = HiringRequirements(skills=["java"])
    ranked = ranker.rank(
        [(_java_product(), 0.9)], req, boosted_types=set(), excluded_types={"K"}, top_k=5
    )
    assert ranked == []


def test_boosted_type_improves_rank_over_unboosted_equal_semantic_score():
    ranker = RequirementsAwareRanker()
    req = HiringRequirements()
    candidates = [(_java_product(), 0.5), (_opq_product(), 0.5)]
    ranked = ranker.rank(candidates, req, boosted_types={"P"}, excluded_types=set(), top_k=5)
    assert ranked[0].product.name == "OPQ32r"


def test_breakdown_contains_all_explainability_factors():
    ranker = RequirementsAwareRanker()
    req = HiringRequirements(skills=["java"], seniority="senior")
    ranked = ranker.rank([(_java_product(), 0.7)], req, boosted_types=set(), excluded_types=set(), top_k=5)
    breakdown = ranked[0].breakdown
    for key in ("semantic_similarity", "skill_overlap", "level_match", "type_match", "business_rule", "final_score"):
        assert key in breakdown


def test_top_k_is_respected():
    ranker = RequirementsAwareRanker()
    req = HiringRequirements()
    candidates = [(_java_product(), 0.9), (_opq_product(), 0.8)]
    ranked = ranker.rank(candidates, req, boosted_types=set(), excluded_types=set(), top_k=1)
    assert len(ranked) == 1
