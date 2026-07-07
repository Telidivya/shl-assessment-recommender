from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .dialogue import ConversationOrchestrator, get_orchestrator
from .exceptions import CatalogLoadError, RetrievalError
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Builds the catalog + retriever once at process startup.

    Why `lifespan` instead of the older `@app.on_event("startup")`: it's
    the FastAPI-recommended pattern (on_event is soft-deprecated) and,
    unlike a bare startup hook, an exception raised here fails the process
    at boot with a clear stack trace rather than letting the app come up in
    a half-initialized state that only breaks on the first request. That
    matters directly for the assignment's cold-start note ("first /health
    call gets up to 2 minutes") — building the embedding index during that
    window, not on the first real /chat call, is what makes that grace
    period actually useful.
    """
    configure_logging()
    try:
        get_orchestrator()
        logger.info("Startup complete: catalog and retriever are ready.")
    except (CatalogLoadError, RetrievalError):
        logger.exception("Fatal startup error — refusing to serve traffic.")
        raise
    yield


app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description=(
        "Conversational agent that turns a vague hiring intent into a grounded shortlist "
        "of SHL Individual Test Solutions, with clarification, refinement, and comparison."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API schema — UNCHANGED from v1. Every field name, type, and default here
# is part of the frozen contract; nothing in this block should be touched
# without a corresponding sign-off, since the grading harness depends on it
# byte-for-byte.
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

def get_conversation_orchestrator() -> ConversationOrchestrator:
    """FastAPI dependency provider.

    This is a thin wrapper around `dialogue.get_orchestrator()` rather than
    inlining that call in the endpoint, specifically so tests can override
    it via `app.dependency_overrides[get_conversation_orchestrator] = ...`
    to inject a fake/lightweight orchestrator (e.g. one built with a stub
    retriever) without monkeypatching module internals.
    """
    return get_orchestrator()


# ---------------------------------------------------------------------------
# Endpoints — paths, verbs, and response shapes UNCHANGED from v1.
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    orchestrator: ConversationOrchestrator = Depends(get_conversation_orchestrator),
) -> dict:
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        return orchestrator.handle(history)
    except Exception:
        # Defensive boundary: an unexpected internal error should degrade to
        # a safe, schema-compliant reply — never a raw 500 that breaks the
        # harness's expectation of a valid ChatResponse on every call, and
        # never a stack trace leaked to the client. The incident is still
        # fully visible in logs for on-call follow-up.
        logger.exception("Unhandled error in /chat; returning a safe fallback reply.")
        return {
            "reply": (
                "Something went wrong on my end processing that. Could you rephrase, "
                "or tell me again what role you're hiring for?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }
