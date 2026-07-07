import numpy as np

from app.catalog import Product
from app.retrieval import (
    BruteForceCosineIndex,
    SemanticRetriever,
    TfidfEmbeddingBackend,
    _catalog_fingerprint,
    _l2_normalize,
)


def _sample_products() -> list[Product]:
    return [
        Product(name="Java 8 (New)", url="https://www.shl.com/x/java/", test_type=["K"],
                description="Knowledge test on Java programming."),
        Product(name="Occupational Personality Questionnaire OPQ32r",
                url="https://www.shl.com/x/opq/", test_type=["P"],
                description="Personality assessment measuring workplace behavioral style."),
        Product(name="Verify - Numerical Ability", url="https://www.shl.com/x/verify/",
                test_type=["A"], description="Cognitive ability test measuring numerical reasoning."),
    ]


def test_retriever_ranks_relevant_product_first():
    retriever = SemanticRetriever(TfidfEmbeddingBackend(), BruteForceCosineIndex, cache_path=None)
    retriever.build(_sample_products())
    results = retriever.retrieve("Java programming knowledge test", top_n=3)
    assert results, "expected at least one candidate"
    top_product, top_score = results[0]
    assert top_product.name == "Java 8 (New)"
    assert top_score > 0


def test_retriever_returns_empty_list_for_empty_query():
    retriever = SemanticRetriever(TfidfEmbeddingBackend(), BruteForceCosineIndex, cache_path=None)
    retriever.build(_sample_products())
    assert retriever.retrieve("   ", top_n=5) == []


def test_retriever_top_n_is_respected():
    retriever = SemanticRetriever(TfidfEmbeddingBackend(), BruteForceCosineIndex, cache_path=None)
    retriever.build(_sample_products())
    results = retriever.retrieve("personality assessment", top_n=1)
    assert len(results) <= 1


def test_l2_normalize_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
    normalized = _l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0)


def test_catalog_fingerprint_changes_when_products_change():
    products_a = _sample_products()
    products_b = _sample_products()
    products_b[0].description = "A completely different description."
    fp_a = _catalog_fingerprint(products_a, "backend-x")
    fp_b = _catalog_fingerprint(products_b, "backend-x")
    assert fp_a != fp_b


def test_catalog_fingerprint_changes_when_backend_changes():
    products = _sample_products()
    fp_a = _catalog_fingerprint(products, "backend-x")
    fp_b = _catalog_fingerprint(products, "backend-y")
    assert fp_a != fp_b


def test_embedding_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.npz"
    retriever = SemanticRetriever(TfidfEmbeddingBackend(), BruteForceCosineIndex, cache_path=cache_path)
    retriever.build(_sample_products())
    assert cache_path.exists()

    # A second retriever pointed at the same cache file should load vectors
    # from disk rather than recomputing (Priority 9) -- observable here as
    # "doesn't raise and still returns sane results", since the two backend
    # instances are independent TfidfVectorizer fits.
    retriever2 = SemanticRetriever(TfidfEmbeddingBackend(), BruteForceCosineIndex, cache_path=cache_path)
    retriever2.build(_sample_products())
    results = retriever2.retrieve("Java programming", top_n=1)
    assert results[0][0].name == "Java 8 (New)"
