"""Custom exceptions for the SHL assessment agent.

A dedicated hierarchy (instead of raising/catching bare `Exception` or
built-ins like `ValueError` everywhere) means callers can distinguish
*what kind* of failure occurred without parsing error strings, and
`main.py`'s exception handlers can map each kind to the right HTTP/response
behavior deliberately rather than by accident.
"""
from __future__ import annotations


class ShlAgentError(Exception):
    """Base class for all errors raised by this service."""


class CatalogLoadError(ShlAgentError):
    """Raised when the catalog file is missing, malformed, or empty.

    This is intentionally fatal at startup (see `main.py`'s startup hook):
    a service that silently serves an empty or corrupt catalog would violate
    the "recommendations only from the SHL catalog" requirement in the worst
    possible way — by having nothing grounded to recommend from at all.
    """


class RetrievalError(ShlAgentError):
    """Raised when the semantic retriever fails to build or query.

    Distinct from `CatalogLoadError` because retrieval failures are
    *recoverable* at request time — `dialogue.py` catches this and falls
    back to the deterministic keyword search rather than failing the
    request outright.
    """
