"""
Semantic retrieval over the SHL catalog.

Priority 1 asks for: SentenceTransformers (all-MiniLM-L6-v2) embeddings, a
FAISS index, Top-20 retrieval feeding a Top-5 ranking stage. That's the
primary path implemented here (`SentenceTransformerBackend` +
`FaissVectorIndex`).

Both dependencies are optional at import time, behind small `Protocol`
interfaces (`EmbeddingBackend`, `VectorIndex`) — this is a Dependency
Inversion boundary, not a hedge: `SemanticRetriever` only ever talks to the
protocol, never to `sentence_transformers` or `faiss` directly, so swapping
either implementation (e.g. a different model, a different ANN library)
touches one factory function and nothing else.

Why a fallback exists at all: this is a hiring-facing production service. A
model registry outage, a locked-down container with no egress to download
model weights, or a missing optional dependency should degrade retrieval
quality, not take `/chat` down. `build_default_retriever()` tries the real
stack first and falls back to a TF-IDF backend + brute-force cosine index
(both pure-Python/numpy/sklearn, no network, no native extension) only if
the real stack can't be constructed — logging loudly when it does.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .catalog import Catalog, Product

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_N_RETRIEVE = 20  # Priority 1: retrieve top-20 candidates semantically
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"


# ---------------------------------------------------------------------------
# Protocols (Dependency Inversion boundary)
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 matrix of embeddings for `texts`."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    def build(self, vectors: np.ndarray) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) of the top_k nearest vectors."""
        ...


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class SentenceTransformerBackend:
    """Primary embedding backend: all-MiniLM-L6-v2 via sentence-transformers."""

    name = f"sentence-transformers:{EMBEDDING_MODEL_NAME}"

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None  # lazy: don't pay model-load cost at import time

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: optional dep
            logger.info("Loading SentenceTransformer model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.astype("float32")


class TfidfEmbeddingBackend:
    """Deterministic fallback embedding backend requiring only scikit-learn.

    Not a drop-in semantic replacement for MiniLM, but it preserves the same
    contract (text -> fixed-size vector, cosine-comparable) so every
    downstream component (index, cache, ranker) is unaffected by which
    backend is active. Fit once on the catalog corpus at build time.
    """

    name = "tfidf-fallback"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  # local import
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            # Queries can arrive before/without an explicit fit call in edge
            # cases (e.g. unit tests exercising encode() directly); fit on
            # whatever we're given rather than raising.
            self.fit(texts)
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype("float32")


# ---------------------------------------------------------------------------
# Vector indexes
# ---------------------------------------------------------------------------

class FaissVectorIndex:
    """Primary ANN index: FAISS flat inner-product index (cosine, since
    vectors are L2-normalized before insertion)."""

    def __init__(self, dim: int):
        import faiss  # local import: optional dep
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)

    def build(self, vectors: np.ndarray) -> None:
        self._index.reset()
        self._index.add(np.ascontiguousarray(vectors, dtype="float32"))

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, indices = self._index.search(
            np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32"), top_k
        )
        return scores[0], indices[0]


class BruteForceCosineIndex:
    """Fallback index: exact cosine similarity via a matrix multiply.

    At catalog scale (hundreds to low thousands of items) this is not a
    meaningful performance compromise — FAISS earns its keep at far larger
    scale, so this fallback is a correctness-preserving, dependency-free
    substitute, not a degraded approximation.
    """

    def __init__(self, dim: int):
        self._dim = dim
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self._vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None or len(self._vectors) == 0:
            return np.array([]), np.array([])
        sims = self._vectors @ query_vector
        top_k = min(top_k, len(sims))
        top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return sims[top_idx], top_idx


# ---------------------------------------------------------------------------
# Embedding cache (Priority 9: avoid recomputing embeddings on every restart)
# ---------------------------------------------------------------------------

def _catalog_fingerprint(products: list[Product], backend_name: str) -> str:
    """Hash of (catalog content + backend identity) used as a cache key.

    Recomputes embeddings only when the catalog changes or the backend
    changes — not on every process restart, which matters for cold-start
    latency in the "first /health call gets 2 minutes" hosting scenario
    called out in the assignment.
    """
    payload = json.dumps(
        [[p.name, p.url, p.description, p.test_type] for p in products],
        sort_keys=True,
    ) + backend_name
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _EmbeddingCache:
    fingerprint: str
    vectors: np.ndarray

    @classmethod
    def load(cls, path: Path, expected_fingerprint: str) -> "_EmbeddingCache | None":
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                fingerprint = str(data["fingerprint"])
                if fingerprint != expected_fingerprint:
                    return None
                return cls(fingerprint=fingerprint, vectors=data["vectors"])
        except Exception:  # pragma: no cover - corrupt cache is non-fatal
            logger.warning("Embedding cache at %s unreadable; recomputing.", path)
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, fingerprint=self.fingerprint, vectors=self.vectors)


# ---------------------------------------------------------------------------
# Semantic retriever
# ---------------------------------------------------------------------------

class SemanticRetriever:
    """Owns the embedding backend + vector index lifecycle for one catalog."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        index_factory,
        cache_path: Path | None = None,
    ):
        self._backend = embedding_backend
        self._index_factory = index_factory
        self._cache_path = cache_path
        self._index: VectorIndex | None = None
        self._products: list[Product] = []

    def build(self, products: list[Product]) -> None:
        self._products = products
        texts = [f"{p.name}. {p.description}" for p in products]

        if hasattr(self._backend, "fit"):
            self._backend.fit(texts)  # TF-IDF-style backends need a fit pass

        vectors = self._load_or_compute_vectors(texts)
        vectors = _l2_normalize(vectors)

        self._index = self._index_factory(vectors.shape[1])
        self._index.build(vectors)
        logger.info(
            "SemanticRetriever built: backend=%s products=%d dim=%d",
            self._backend.name, len(products), vectors.shape[1],
        )

    def _load_or_compute_vectors(self, texts: list[str]) -> np.ndarray:
        if self._cache_path is not None:
            fingerprint = _catalog_fingerprint(self._products, self._backend.name)
            cached = _EmbeddingCache.load(self._cache_path, fingerprint)
            if cached is not None:
                logger.info("Loaded cached embeddings (%s)", self._cache_path.name)
                return cached.vectors
            vectors = self._backend.encode(texts)
            _EmbeddingCache(fingerprint=fingerprint, vectors=vectors).save(self._cache_path)
            return vectors
        return self._backend.encode(texts)

    def retrieve(self, query_text: str, top_n: int = TOP_N_RETRIEVE) -> list[tuple[Product, float]]:
        if not query_text.strip() or self._index is None or not self._products:
            return []
        query_vec = _l2_normalize(self._backend.encode([query_text]))[0]
        scores, indices = self._index.search(query_vec, min(top_n, len(self._products)))
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._products):
                continue
            results.append((self._products[int(idx)], float(score)))
        return results


def build_default_retriever(catalog: Catalog) -> SemanticRetriever:
    """Factory: real SentenceTransformer+FAISS stack, degrading gracefully.

    This is the one place that decides which concrete backend/index to use.
    Everything else in the codebase depends on the `SemanticRetriever`
    abstraction, so this function is the entire blast radius of a dependency
    swap or a missing-package incident.
    """
    cache_path = CACHE_DIR / "catalog_embeddings.npz"

    try:
        backend: EmbeddingBackend = SentenceTransformerBackend()
        # Touch the model now so a missing/undownloadable model fails fast,
        # here, with a clear log line — not silently on the first request.
        backend.encode(["healthcheck"])
        index_factory = FaissVectorIndex
        _ = __import__("faiss")  # confirm FAISS import succeeds before committing
        logger.info("Using SentenceTransformer + FAISS retrieval stack.")
    except Exception as exc:  # noqa: BLE001 - intentionally broad: any failure degrades, doesn't crash
        logger.warning(
            "Falling back to TF-IDF + brute-force retrieval (reason: %s). "
            "Install sentence-transformers and faiss-cpu, with network access "
            "to download model weights, for full semantic retrieval quality.",
            exc,
        )
        backend = TfidfEmbeddingBackend()
        index_factory = BruteForceCosineIndex
        cache_path = CACHE_DIR / "catalog_embeddings_tfidf.npz"

    retriever = SemanticRetriever(backend, index_factory, cache_path=cache_path)
    retriever.build(catalog.products)
    return retriever
