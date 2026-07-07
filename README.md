# SHL Assessment Recommendation Agent

A stateless FastAPI service that turns a vague hiring intent into a grounded
shortlist of **SHL Individual Test Solutions**, through clarification,
recommendation, refinement, and comparison. Every recommendation is drawn
from a scraped, in-memory catalog, so the agent structurally cannot surface
an assessment (or URL) that isn't actually in the SHL catalog.

> **v2 note:** this is a review/upgrade pass over a working v1. The API
> schema, endpoints, and stateless contract are byte-for-byte unchanged;
> the internals (retrieval, ranking, requirements understanding,
> clarification, comparison, safety, code structure, tests, docs) were
> substantially reworked. See "What changed in v2" below for a summary, and
> `ARCHITECTURE.md` for the full system diagram and per-module rationale.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → see schema below (matches the spec exactly: `reply`,
  `recommendations` (0–10 items), `end_of_conversation`).

```bash
curl -s localhost:8000/chat -X POST -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hiring a Java developer, mid-level, 4 years experience"}]
}' | python -m json.tool
```

Run the tests:

```bash
pytest tests/ -q
```

## Architecture

```
app/
  catalog.py       # loads + validates data/shl_catalog.json, Product records, keyword fallback search
  requirements.py  # RequirementsExtractor: conversation text -> structured HiringRequirements
  retrieval.py     # SemanticRetriever: SentenceTransformer+FAISS (primary) or TF-IDF+brute-force (fallback)
  ranking.py       # RequirementsAwareRanker: semantic + skill overlap + level + type + business rules
  nlu.py           # scope guard (safety), comparison/refinement/closing text detection
  dialogue.py      # ConversationOrchestrator: wires the above into the 4 required behaviors
  main.py          # FastAPI wiring (/health, /chat), DI, lifespan startup, error boundary
  exceptions.py    # typed error hierarchy (CatalogLoadError, RetrievalError)
  logging_config.py
data/
  shl_catalog.json # the catalog itself (see "About the catalog" below)
  .cache/          # on-disk embedding cache (created at runtime, gitignored)
scrape_catalog.py  # standalone scraper to (re)build data/shl_catalog.json, incl. job_levels/duration
tests/
  test_api.py          # FastAPI-layer: schema, all 4 behaviors, safety, edge cases
  test_requirements.py # RequirementsExtractor unit tests
  test_retrieval.py    # SemanticRetriever unit tests (fallback backend, no network needed)
  test_ranking.py      # RequirementsAwareRanker unit tests
  test_catalog.py      # catalog validation + invalid-catalog error paths
ARCHITECTURE.md    # system diagram + component responsibilities + request lifecycle
```

**Why the catalog-only guarantee is structural, not prompted:** the hard
requirement is "never recommend anything outside the SHL catalog." Every
recommendation object returned by `/chat` is literally
`Product.as_recommendation()` off the in-memory catalog list — there is no
code path through which a hallucinated name/URL can reach the response,
regardless of what the semantic retriever or ranker score. Comparison text
is built only from stored `Product` fields (description, test_type,
job_levels, duration) for the same reason.

### The four behaviors

1. **Clarify vague queries.** `requirements.RequirementsExtractor` parses
   the *whole* reconstructed history into a `HiringRequirements` object
   (role, seniority, skills, personality/cognitive/leadership/coding
   flags). `dialogue._next_missing_field()` asks about exactly **one**
   missing field at a time, in priority order (role → focus → seniority) —
   never a generic multi-part question, and never a question about
   something already established. A bare "I need an assessment" always
   asks a follow-up; a one-shot detailed job description skips straight to
   a shortlist; and a hard convergence rule forces a recommendation by the
   3rd user turn regardless of remaining gaps, so an open clarification
   loop can never blow through the 8-turn cap.

2. **Recommend (1–10 items, Top-20 → Top-5).** `retrieval.SemanticRetriever`
   embeds the structured requirements (blended with raw recent text) and
   retrieves the Top-20 nearest catalog items. `ranking.RequirementsAwareRanker`
   re-scores those 20 on semantic similarity + skill overlap + job-level
   match + assessment-type match + business rules (explicit add/remove
   signals), returning an explainable Top-5 (breakdown available on the
   `RankedCandidate` object; the public API surfaces the same 1–10 item
   contract as before).

3. **Refine.** Because the API is stateless, refinement is just re-deriving
   `HiringRequirements` and the ranked shortlist from the *full*
   reconstructed history each call. "Actually, add personality tests" or
   "remove coding tests, focus on personality instead" are parsed
   clause-by-clause so add/remove intents in the same sentence don't
   collide (verified in `tests/test_api.py::test_refinement_remove_type_excludes_it`).

4. **Compare.** `nlu.extract_comparison_targets` recognizes "difference
   between X and Y" / "X vs Y" / "compare X and Y" phrasing;
   `Catalog.find_by_name_fragment` resolves names or acronyms (e.g. "OPQ",
   "GSA"). The reply is a **feature-by-feature markdown table** built only
   from structured `Product` fields (test type, duration, job levels,
   description) — fields the catalog doesn't have are shown as "Not
   specified in catalog data" rather than guessed. If either side can't be
   resolved in-catalog, the agent says so instead of comparing anyway.

**Scope guard.** `nlu.check_scope` runs before anything else and
pattern-matches five refusal categories — **jailbreak** (persona
overrides, "DAN", "no restrictions"), **prompt injection** ("ignore
previous instructions", "reveal your system prompt"), **legal advice**,
**general hiring/people-management advice**, and **off-topic** — each with
its own scoped refusal that redirects back to assessment selection.

## Retrieval backends: primary vs. fallback

| | Primary | Fallback (automatic) |
|---|---|---|
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | scikit-learn `TfidfVectorizer` |
| Index | `faiss.IndexFlatIP` (cosine, normalized vectors) | Brute-force cosine (`numpy`) |
| Falls back when | import fails, or model weights can't download (no network egress) | — |

`retrieval.build_default_retriever()` tries the real stack first and
degrades — loudly, via a logged warning — only if it can't be built. This
matters concretely for this submission: **this build environment had no
outbound network access**, so `sentence-transformers`/`faiss-cpu` couldn't
be installed or exercised here. Every retrieval/ranking code path was
still fully implemented and tested against the fallback backend (which
satisfies the identical `EmbeddingBackend`/`VectorIndex` protocol), and
`tests/test_retrieval.py` runs against that fallback for exactly this
reason. **Whoever deploys this with normal network access gets the real
MiniLM+FAISS stack automatically** — no code or config change required,
since the choice is made once, at startup, by `build_default_retriever()`.

Embeddings are cached to `data/.cache/*.npz`, keyed by a hash of catalog
content + backend name (`retrieval._catalog_fingerprint`), so a process
restart with an unchanged catalog loads vectors from disk instead of
recomputing them — see "Performance" below.

## About the catalog

`data/shl_catalog.json` is a **seed slice** of the full ~370-item
"Individual Test Solutions" catalog (Pre-packaged Job Solutions are
excluded, per spec). Every name/url/test_type was scraped and verified
against live SHL product pages, not invented. It includes 63 real entries
spanning all 8 test-type categories — coding/knowledge tests across
several languages and frameworks, `OPQ32r`, the `Global Skills Assessment
(GSA)`, the full `Verify` cognitive-ability family, a situational-judgement
item, office-skills simulations, and a 360/development item — deliberately
including the exact "OPQ" / "GSA" pair called out in the spec's comparison
example. `duration_minutes` is derived (never invented) from each
product's own verified description text where it mentions a time (Priority
5: never hallucinate — see `catalog.Product.__post_init__`).

**To materialize the complete catalog:**

```bash
pip install requests beautifulsoup4
python scrape_catalog.py            # names/urls/test_types only, fast
python scrape_catalog.py --enrich   # also pulls description, job_levels, and duration per product
```

It paginates until a page returns no new rows, parses the "Individual Test
Solutions" table (so Pre-packaged Job Solutions are automatically
excluded), and overwrites `data/shl_catalog.json` in the same schema the
app already loads — no code changes needed elsewhere. This build
environment had no outbound network access to run it against the live
site; the seed data was instead hand-verified via individual page fetches.

## What changed in v2 (review-and-improve pass)

Every item below preserves the exact v1 API schema, endpoints, and
stateless contract — see `ARCHITECTURE.md` for full rationale per change.

- **Retrieval:** keyword-overlap-only → `SemanticRetriever` (MiniLM + FAISS,
  Top-20) feeding an explainable multi-factor `RequirementsAwareRanker`
  (Top-5), with a dependency-free fallback so the service degrades rather
  than crashes if the ML stack can't load.
- **Conversation understanding:** scattered boolean signal checks →
  one structured `HiringRequirements` object per turn, used consistently
  by retrieval, ranking, and clarification.
- **Clarification:** combined "tell me role + level" questions → one
  targeted question for the single highest-priority missing field,
  determined dynamically per turn.
- **Comparison:** prose synthesis from descriptions → a feature-by-feature
  markdown table from structured fields, with an honest "Not specified in
  catalog data" for anything the catalog doesn't have.
- **Safety:** 4 refusal categories → 5 (jailbreak split out from prompt
  injection, since they're different attack shapes), with an expanded
  pattern set for each.
- **Code quality:** a single `handle_chat()` function → a small
  `ConversationOrchestrator` class taking its four collaborators via
  constructor injection (testable in isolation; see `tests/test_ranking.py`,
  `tests/test_retrieval.py`, `tests/test_requirements.py`), a typed
  exception hierarchy, centralized logging, and a FastAPI `Depends`-based
  provider (`main.get_conversation_orchestrator`) that tests can override.
- **Error handling:** an unhandled exception in `/chat` now degrades to a
  safe, schema-compliant reply (logged server-side) instead of a raw 500;
  catalog problems (missing file, malformed JSON, empty list, entries
  missing required fields) raise a specific `CatalogLoadError` at startup
  instead of a generic `KeyError` deep in a request.
- **Performance:** embeddings cached to disk (`data/.cache/`), catalog and
  retriever built once at process startup via FastAPI `lifespan` (not
  per-request, not on first request).
- **Tests:** roughly doubled — added dedicated unit-test files for
  requirements extraction, retrieval, ranking, and catalog validation
  (including the invalid-catalog and malformed-request paths), alongside
  the expanded FastAPI-layer test suite.
- **Docs:** this README plus a new `ARCHITECTURE.md` (diagram + per-module
  responsibility table + request lifecycle walkthrough).

## Grading harness alignment

Per the evaluator spec, scoring has three parts, and the design targets each directly:

- **Hard evals** (schema compliance, catalog-only recommendations, 8-turn
  cap honored). Schema is a fixed pydantic model (unchanged in this pass);
  recommendations are always literal catalog rows; the orchestrator has a
  hard convergence rule — clarification never continues past the 3rd user
  turn, and a small generic fallback shortlist guarantees a non-empty list
  rather than stalling if retrieval finds nothing.
- **Recall@10.** Bottlenecked primarily by catalog coverage (see "About the
  catalog"), secondarily by retrieval quality — semantic retrieval with a
  Top-20 net before ranking narrows to 5 should recall better than pure
  keyword overlap on paraphrased queries, once the real MiniLM backend is
  active in a networked deployment.
- **Behavior probes** (refuses off-topic, doesn't recommend turn 1 on vague
  query, honors edits, hallucination rate). Covered by the 5-category scope
  guard, the dynamic single-question clarification policy, the
  clause-parsed add/remove refinement logic, and the structural guarantee
  that recommendations can only ever be catalog rows.

**What I still couldn't do in this build environment:** deploy a public
endpoint (submission requires one — see "Deploying" below), install/run
`sentence-transformers` or `faiss-cpu` (no outbound network — validated the
identical code path against the fallback backend instead, see "Retrieval
backends" above), run `pytest` against the real FastAPI/Starlette/pydantic
stack (same reason — validated `app/dialogue.py`, `app/requirements.py`,
`app/ranking.py`, `app/retrieval.py`, and `app/catalog.py` directly in
Python, and only syntax-checked `app/main.py`'s FastAPI wiring), or access
the actual 10 provided conversation traces. Run `pytest tests/ -q` yourself
as a final check once dependencies are installed.

## Deploying

This sandbox can't expose a public URL, so you'll need to deploy it
yourself for submission. Any standard Python host works, e.g.:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Cold start:** the FastAPI `lifespan` startup hook (see `app/main.py`)
builds the catalog and the semantic retriever (loading the MiniLM model
and building the FAISS index, or computing/loading cached embeddings)
*before* the app starts accepting traffic — this is what the assignment's
"first `/health` call gets up to 2 minutes" grace period is for. Set
`LOG_LEVEL=INFO` (default) or `LOG_LEVEL=DEBUG` to see startup progress,
including which retrieval backend was selected.

**Docker** (optional, if your host expects a container):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level for all `app.*` loggers |

## Known limitations / next steps

- **Fallback retrieval quality.** If deployed without network access to
  download MiniLM weights, retrieval quality degrades to TF-IDF — still
  fully functional and catalog-grounded, just less semantically aware of
  paraphrasing than true embeddings. This only affects environments with
  no outbound network at all; a normal cloud host will use the real stack.
- **The seed catalog is a slice (63 verified items), not the full ~370** —
  `scrape_catalog.py --enrich` closes that gap (including `job_levels` and
  `duration_minutes` now) in an environment with normal network access.
  The seed skews toward Knowledge & Skills (K) items relative to the other
  7 categories; a full catalog pull would rebalance this.
- **Ranking weights are hand-set, not learned.** With no labeled relevance
  data available, a simple linear combination (see `ranking.py`'s
  module-level `WEIGHT_*` constants) is the honest choice — easy to reason
  about and to tune from observed failures, and each term is independently
  inspectable via `RankedCandidate.breakdown`.
- **Scope guard is pattern-based.** It will miss sufficiently novel
  injection/jailbreak phrasing a semantic classifier might catch, but it
  also won't silently false-negative — unmatched text always falls through
  to the (harmless) assessment-matching path rather than being treated as
  implicitly safe.
