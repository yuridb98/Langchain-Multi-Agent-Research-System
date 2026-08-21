# Architecture & Design Notes

This document is the deep-dive companion to the README. The README tells you how to run the
project; this file explains how it's put together and why, in enough detail to answer follow-up
questions about it (in an interview, a code review, or six months from now when you've forgotten
the details yourself).

---

## 1. What this actually is

A four-agent research pipeline: give it a topic, it searches the web, reads the most relevant
result in depth, writes a structured report, and has a second model critique that report. It's
exposed two ways — a Streamlit UI and a CLI — both calling the same orchestration code.

It is deliberately **not** a general agent framework, not a RAG system, not multi-turn chat. The
scope is narrow on purpose: one linear pipeline, four fixed roles, no persistence between runs.
That narrowness is what makes it possible to reason about the whole system at once, which matters
more for a portfolio project than breadth does — a reviewer can read every file in twenty
minutes and understand exactly what happens on every request.

## 2. The pipeline, concretely

```
topic
  │
  ▼
┌─────────────┐   web_search (Tavily)         ┌─────────────┐
│ Search Agent│──────────────────────────────▶│ search text │
│ (ReAct)     │                                └─────────────┘
└─────────────┘                                       │
                                                        ▼
┌─────────────┐   scrape_url (requests +      ┌─────────────┐
│ Reader Agent│   trafilatura/readability/bs4) │reader text  │
│ (ReAct)     │◀───── picks a URL from ────────│ (or "" on   │
└─────────────┘        search text             │  failure)   │
       │                                        └─────────────┘
       ▼
┌─────────────┐
│ Writer Chain│──▶ report (markdown)
│ (prompt|llm)│
└─────────────┘
       │
       ▼
┌─────────────┐
│ Critic Chain│──▶ feedback (score + notes)
│ (prompt|llm)│
└─────────────┘
```

Search and Reader are **LangGraph ReAct agents** (`langchain.agents.create_agent`) — they get a
system prompt, one tool each, and loop tool-call → observation → tool-call until they decide
they're done, then return their final message. Writer and Critic are plain **LCEL chains**
(`prompt | llm | StrOutputParser()`) — one prompt, one model call, no tool use, no looping. That
split isn't arbitrary: Search and Reader need to *decide* something (which query to run, which
URL to pick) and act on a tool, which is what the ReAct loop is for. Writer and Critic are pure
transformations of text they're already handed — wiring them through a graph would add nothing.

Each step's output feeds the next step's input as plain Python values passed through a
dataclass (`ResearchResult`), not through shared agent memory or a graph state object. There's no
LangGraph orchestration *across* the four steps — the pipeline itself is a straight-line Python
function (`run_research_pipeline` in `src/pipeline.py`). Using LangGraph for the outer
loop too was considered and rejected: four sequential steps with no branching, no retries-with-
different-strategy, and no need to persist/resume mid-run don't get anything from a graph
abstraction. A function that runs top to bottom is easier to read, easier to debug, and doesn't
require learning LangGraph state schemas to understand what order things happen in.

## 3. Why LangChain/LangGraph, Streamlit, Tavily

- **LangChain (`create_agent`) + LangGraph** — chosen because the ReAct-agent-with-tools pattern
  is exactly what LangChain's prebuilt agent constructor does out of the box, and it's the most
  recognizable "I know how agentic LLM apps work" signal in this ecosystem. Rolling a custom
  tool-calling loop by hand would be more code for no behavioral difference.
- **Streamlit** — the fastest way to put a real, usable UI in front of a Python backend with no
  separate frontend build step. The tradeoff (documented below) is Streamlit's execution model:
  the whole script re-runs top to bottom on every interaction, which shapes a lot of the code
  (see §6, session state).
- **Tavily** — a search API built specifically for LLM agents (returns clean snippets instead of
  raw SERPs you'd have to parse yourself). Swapping it for another provider only touches
  the `web_search` function in `src/tools.py`.
- **OpenAI (via `langchain_openai.ChatOpenAI`)** — single-provider by design for now (see §9 for
  the multi-provider extension path). All model selection is restricted to their cheap tiers
  (nano/mini), because this app is meant to run on the owner's own API key with public or
  semi-public access — see §7.

## 4. Folder structure and what lives where

```
app.py              Streamlit entrypoint — thin, no business logic
main.py             CLI entrypoint — same pipeline, argparse instead of widgets

src/
  config.py          secrets (st.secrets -> env -> .env), runtime limits, cheap-model allowlist
  prompts.py           every prompt template, in one place
  agents.py             make_llm() + the four build_*_agent/build_*_chain factories
  tools.py                web_search (Tavily) + scrape_url (with an SSRF guard)
  pipeline.py              run_research_pipeline() — the orchestration, the only one
  ui.py                     theme CSS + sidebar model controls + results rendering

tests/               offline unit tests — no API keys needed to run them
.streamlit/config.toml   theme tokens (colors), not CSS
.github/workflows/ci.yml lint + test on push/PR
```

Six modules directly under `src/`, no subpackages. Each still only imports the ones "below"
it — `ui` imports `pipeline`, `pipeline` imports `agents` and `config`, `agents` imports `tools`,
`prompts`, and `config`, and `config`/`tools` import nothing else in the project. Nothing in
`tools` or `agents` knows Streamlit exists. That's what let `main.py` and `app.py` end up as two
thin shells over identical logic instead of two divergent implementations (see §10). The
boundary is what matters, not whether it's a folder or a file — see §12 for why this ended up
flat instead of nested into packages.

## 5. Design decisions and the reasoning behind them

**Factories instead of a module-level LLM singleton.** The original code had one
`llm = ChatOpenAI(...)` built at import time and shared by all four agents. That meant (a) every
agent was forced onto the same model, so per-agent selection was structurally impossible, and
(b) a missing API key crashed the entire app the moment Python imported the module — before a
user had done anything. `make_llm(model_id, temperature)` in `src/agents.py` is a factory,
called fresh each time from inside `run_research_pipeline` — i.e., lazily, on an actual run, not
at import. A bad or missing key now surfaces as a caught `ConfigError` shown in the UI, not a
stack trace on page load.

**The model allowlist is enforced in the pipeline, not just the UI.** `resolve_model(agent,
requested_id)` in `src/config.py` silently falls back to a safe default if the requested
id isn't in the allowlist. The UI only ever *offers* allowlisted ids, so in normal use this never
triggers — but it's called again inside `run_research_pipeline` regardless of caller, so a
forged `session_state` value, a bug in the sidebar, or someone hitting the pipeline function
directly can never reach an expensive model. Client-side restrictions are a UX nicety;
server-side restrictions are the actual security boundary. Since this runs on the owner's API
key, that boundary matters.

**One config module, one secret-resolution order.** `src/config.py` is the only place
`load_dotenv()` is called and the only place that reads `st.secrets`. Every other module asks
`require_secret("OPENAI_API_KEY")` and gets back a value or a `ConfigError` — it doesn't know or
care whether that came from Streamlit's secrets panel or a local `.env` file. Order is
`st.secrets` → environment (populated by `.env` via `python-dotenv` locally). That order is what
makes the exact same code run unmodified on Streamlit Community Cloud and on a laptop.

**The pipeline degrades per-step instead of failing all-or-nothing.** Each of the four steps is
wrapped in its own try/except and appends to `result.errors` on failure. If the Reader step
fails (a bad scrape, a timeout), the Writer still runs — on search results alone, producing a
weaker but real report — rather than the whole request dying because one of four steps had a
bad day. Search and Writer failures *do* stop the pipeline early, because there's nothing
meaningful left to hand downstream (no search results means no basis for anything; no report
means nothing for the critic to critique). This is a judgment call, not a rule: it's "fail soft
where partial output is still useful, fail fast where it isn't."

**`on_event(step, status)` callback instead of a return-value-only pipeline.** The pipeline takes
an optional callback fired before/after each step (`"running"` / `"done"` / `"error"`). The
Streamlit UI binds it to `st.status(...).write(...)` for live progress; the CLI binds it to
`print`. This is what makes the *same* orchestration function serve both surfaces with real
progress feedback in each, instead of the UI needing its own copy of the step sequence to know
what to display.

**Native Streamlit components over custom CSS/HTML.** The rewrite replaced ~290 lines of
hand-written CSS and raw-HTML `st.markdown(..., unsafe_allow_html=True)` blocks with
`st.sidebar`, `st.form`, `st.status`, `st.tabs`, `st.container(border=True)`, and theme colors
defined once in `.streamlit/config.toml`. Two concrete reasons: first, every native widget
already handles both light and dark themes correctly, so there's no CSS to maintain against
Streamlit's own updates; second, and more importantly, the old code interpolated raw LLM/scraped
text directly into HTML strings — any `<` in scraped page content would corrupt the layout, and
it's the kind of thing that looks fine in a demo and breaks on the first real topic. Model output
is untrusted text; it goes through `st.markdown(text)` (no `unsafe_allow_html`) or `st.text`, never
string-interpolated into markup.

**SSRF guard on the scraper.** `scrape_url`'s target isn't chosen by the user — it's chosen by
the Reader agent from search results, i.e. from content on the open web. Without a check, a
crafted or compromised page could steer the agent into requesting
`http://169.254.169.254/...` (the cloud metadata endpoint on AWS/GCP/Azure) or an internal
service, and return the response as "scraped content." `scrape_url` in `src/tools.py` resolves the
hostname via `socket.getaddrinfo` and rejects loopback/private/link-local/reserved addresses and
non-http(s) schemes — and re-validates on every redirect hop, since a first check on the
original URL alone would miss a redirect chain that lands somewhere internal.

**Why a dataclass (`ResearchResult`) instead of a plain dict.** Typed fields, IDE
autocomplete, and a place to hang the `.ok` property (`not self.errors`) that both the UI and
CLI use to decide how to render a partially-failed run. It costs nothing and removes an entire
class of "did I spell the key right" bugs that the original dict-based `state = {}` had (the UI
and CLI versions used different key names for the same data before this refactor).

**Tests are offline and mock the LLM boundary.** `tests/test_pipeline.py` replaces
`make_llm`/`build_*` with fakes that return canned strings — it's testing *pipeline behavior*
(does a Reader failure still let the Writer run? does an unrecognized model id get swapped for
the default?), not testing that OpenAI's API works. `tests/test_scraper.py` tests the SSRF guard
against IP literals (no DNS needed, no network needed) and mocks `requests.get` for the
extraction-strategy tests. This means CI runs with zero API keys and zero network calls, which is
both faster and the correct scope for unit tests — an integration test hitting the real OpenAI/
Tavily APIs would be a different, separate test tier, not currently built.

## 6. Streamlit-specific mechanics worth knowing

Streamlit re-executes the entire script top-to-bottom on every user interaction (a click, a
form submit, a widget change). That has two direct consequences in this code:

- **State that must survive a rerun goes in `st.session_state`.** `result` (the last pipeline
  output) and `run_count` (the abuse-guard counter) are the only two session-state keys, both
  initialized once near the top of `app.py`. Anything not in session state resets on every rerun
  — which is fine for the sidebar selectors, since Streamlit widgets keep their own value across
  reruns via their `key` automatically.
- **The submit button's code path runs inline in the same script pass as everything else** —
  there's no separate request handler. That's why `run_research_pipeline` is called directly
  inside `if submitted:` rather than through some background job system. For a single-user or
  low-concurrency deployment this is fine; it would not scale to many concurrent long-running
  requests without moving pipeline execution off the main Streamlit thread (see §9).

`st.status(...)` as a context manager is what gives live step-by-step progress without any
manual rerun/polling logic — the `on_event` callback just calls `.write()` on the open status
box as the pipeline progresses through it synchronously.

## 7. Security & cost posture

This is meant to be deployed with the owner's own API key, possibly to an audience that isn't
just the owner. Everything here follows from that constraint:

- **No key ever reaches the client.** Keys are read server-side only (`src/config.py`),
  never written to `session_state`, never logged (log calls in `src/tools.py`/`src/pipeline.py`
  slice the topic/URL/query they log, never a secret), never rendered.
- **Model allowlist, enforced server-side** (§5) — caps worst-case per-call cost regardless of
  what the client sends.
- **Per-session run counter** (`MAX_RUNS_PER_SESSION`, default 10) — caps total calls from one
  browser session.
- **Bounded output** — `WRITER_MAX_TOKENS` caps the report length; `SCRAPE_CHAR_LIMIT` caps how
  much scraped text ever reaches a prompt.
- **Timeouts and bounded retries** on every outbound call (`REQUEST_TIMEOUT`, `MAX_RETRIES`) so a
  hung request doesn't tie up a session indefinitely.
- **SSRF guard** on the one tool that fetches attacker-influenceable URLs (§5).

What's *not* covered, deliberately out of scope for now: authentication, per-user rate limiting
(the counter is per-session, so a new browser session resets it), and IP-based abuse detection.
See §9 for what closing that gap would take.

## 8. Frameworks/libraries at a glance

| Library | Role | Why this one |
|---|---|---|
| `langchain` / `langchain-core` | prompt templates, `create_agent`, output parsers | the standard for this pattern; `create_agent` gives a correct ReAct loop for free |
| `langgraph` | backs `create_agent`'s compiled agent graph | pulled in transitively by LangChain 1.x's agent constructor |
| `langchain-openai` | `ChatOpenAI` client | official OpenAI integration for LangChain |
| `streamlit` | UI | fastest path to a real UI with no separate frontend |
| `tavily-python` | web search | LLM-oriented search API, clean output |
| `trafilatura`, `readability-lxml`, `beautifulsoup4`, `lxml` | HTML content extraction, cascaded (three strategies, most-specific first) | no single extractor handles every page layout well; falling back through three catches more real-world pages than any one alone |
| `requests` | HTTP fetch for the scraper | simple, synchronous, sufficient for one fetch per run |
| `python-dotenv` | loads `.env` locally | standard for local secret injection |
| `pydantic` (via LangChain) | `SecretStr` for the API key | keeps the key out of any accidental repr/log of the LLM client object |
| `pytest`, `ruff` (dev only) | tests, linting | standard, fast, no config ceremony |

## 9. Known limitations / honest trade-offs

Worth stating plainly rather than glossing over, since these are the questions a good reviewer
will ask:

- **Single LLM provider.** Only OpenAI. LangChain's `init_chat_model` could genericize
  `make_llm` to accept `"openai:gpt-5.4-nano"` / `"anthropic:claude-..."` style ids with fairly
  small changes to `src/agents.py` and `src/config.py` — not done here to keep the scope (and
  the API-key surface to manage) fixed.
- **No persistence.** Every run is stateless; nothing is stored between sessions or users, no
  history, no way to revisit a past report. There's no database in this project at all.
- **No token-level streaming.** Each step is a blocking `.invoke()`; the UI shows step-level
  progress (`st.status`) but not token-by-token output. LangChain supports `.stream()` for this;
  it wasn't added because it meaningfully complicates the pipeline function's control flow (the
  callback becomes a streaming iterator instead of a fire-and-forget event) for a UX gain that
  matters more in a multi-minute chat UI than a four-step report generator.
- **Sequential, not concurrent.** The four steps run one after another, which is a real
  correctness constraint for three of them (Reader needs Search's output, Writer needs both,
  Critic needs the report) — but nothing scrapes multiple URLs in parallel, and a run's total
  latency is the sum of four LLM calls plus one scrape. See §9 extension notes.
- **No evaluation harness.** Report quality is entirely dependent on prompt engineering and
  model choice; there's no automated check that a report is actually good, no regression testing
  against known topics, no LLM-as-judge scoring beyond the built-in Critic step (which is itself
  unvalidated — nothing checks *its* output is sound).
- **No observability beyond local logs.** Standard library `logging`, configured once in
  `app.py`/`main.py`, is all there is; no tracing (LangSmith or otherwise), no per-run cost
  tracking, no dashboards.
- **Session-scoped abuse guard, not identity-scoped.** `MAX_RUNS_PER_SESSION` resets on a new
  browser session — it slows down casual overuse, it doesn't stop a determined actor.

## 10. What this replaced (for context, not action)

The pre-refactor version had all model instantiation as one hardcoded, invalid model string in a
module-level singleton; `app.py` re-implemented the pipeline inline instead of calling
`pipeline.py` (and the two had already drifted — the UI copy had a `"content "` typo, trailing
space and all, that silently broke the Reader step); results were rendered by interpolating raw
model output into HTML strings; and there was no config layer, no tests, no CI, no pinned
dependency versions. None of that is present anymore, but it's useful context for explaining
*why* the current structure looks the way it does — most of the architectural decisions above
are direct responses to a specific failure mode that existed before.

## 11. If you want to extend it

**Add a new tool to an existing agent** — write a `@tool`-decorated function in `src/tools.py`,
add it to the `tools=[...]` list in the relevant `build_*_agent` call in `src/agents.py`.
The agent's system prompt in `src/prompts.py` should mention when to use it.

**Add a fifth pipeline step** — add a field to `ResearchResult`, add a try/except block in
`run_research_pipeline` following the existing pattern, add its label to `STEP_LABELS` in
`src/ui.py`. The `on_event` plumbing needs no changes — it's already generic over step names.

**Add another model provider** — see §9's `init_chat_model` note. The allowlist in `src/config.py`
would need provider-prefixed ids and `make_llm` would parse the prefix.

## 12. A pass at de-generating it: flattening the package structure

An earlier version of this codebase had more files and more package nesting than a four-agent,
one-pipeline project actually needs — six subpackages, six empty `__init__.py` files, nineteen
source files for what's fundamentally 4 agents + 1 pipeline + 2 tools + a UI. It read that way,
too: no one growing this organically from the original three-file version would land on that
structure by accident. More files isn't more scalable; it's just more files. Real scalability
comes from where you draw the boundaries (UI doesn't know about LLMs, tools don't know about
Streamlit, tests can mock at the pipeline/agent seam), not from how many folders those boundaries
live in. This section records what was cut and, more importantly, what was deliberately kept.

**Flattened every subpackage into a single module.** `config/`, `core/`, `agents/`, `pipeline/`,
`tools/`, and `ui/` — each a folder plus an empty `__init__.py` wrapping one to three small files
— became six flat modules directly under `src/` (the tree in §4): `config.py`, `prompts.py`,
`agents.py`, `tools.py`, `pipeline.py`, `ui.py`. Six files instead of nineteen. Every import
still reads the same shape (`from .config import require_secret`) — only whether "depends on"
crosses a folder boundary changed, not which module depends on which.

**Deleted the dead abstraction.** The old `core/errors.py` defined `AppError`, `ConfigError`, and
`PipelineError`. Only `ConfigError` was ever raised or caught anywhere — `PipelineError` was
defined and never used, `AppError` existed only as a base class nothing caught directly
(`grep -rn PipelineError src/` now returns nothing at all — it's gone). `ConfigError` now lives
as a five-line `class ConfigError(Exception)` at the top of `config.py`, the only module that
raises it.

**Cut the logging wrapper.** The old `core/logging.py` was 46 lines wrapping
`logging.basicConfig()` and a `truncate()` helper — standard library `logging` already does what
the wrapper did. `logging.basicConfig(...)` is now called once each in `app.py` and `main.py`
(library code doesn't configure logging; only entrypoints do), and every module that logs uses
`logging.getLogger(__name__)` directly. The truncate-before-logging behavior is now an inline
`text[:200]` at the two or three call sites that need it, not a named helper for a one-line slice.

**Dropped the `@lru_cache` on `make_llm`.** Constructing a `ChatOpenAI` client is cheap and
happens at most four times per pipeline run; caching it by `(model_id, temperature)` was the kind
of optimization that looks deliberate but wasn't solving a real cost. `make_llm` now constructs
directly, no decorator. (The Tavily client in `tools.py` keeps a lazy singleton — `_tavily()` —
since that one genuinely avoids reconstructing a client with the same fixed config across many
tool calls in a long-lived Streamlit process; not every cache is the same kind of unnecessary.)

**Trimmed the module docstrings.** Every file used to open with a paragraph narrating what it
replaced and why ("Replaces the old module-level `llm` singleton...", "Previously this
orchestration existed in two places..."). That history is real and worth keeping — which is why
it now lives in §10 of this document instead — but repeated at the top of every file it read as
generated narration rather than something written under normal time pressure. Docstrings are now
one to three lines, and the "originally / previously / replaces" framing is gone from the source
entirely.

**What was deliberately kept**, because these boundaries do real work, not ceremony: the
UI → pipeline → agents → tools layering (§4's "only import downward" rule — this is what lets
tests mock at the LLM boundary and keeps `app.py` thin); the model allowlist being re-checked
inside the pipeline rather than trusted from the UI (§5); the SSRF guard in `tools.py`; the
`on_event` callback (it's the one piece of indirection that lets the CLI and the UI share one
pipeline with real progress reporting in both, which is a harder problem than it looks and worth
the one small abstraction it costs); the `ResearchResult` dataclass; and the test suite, unchanged
in coverage and now updated only to import from the flat modules. None of that was what made the
repo feel over-generated — file count and docstring verbosity were, and both are fixed.

## 13. Making this stronger for a portfolio/CV

The project is a clean, honest demonstration of agentic-LLM-app engineering: multi-agent
orchestration, tool use, a real (if narrow) security posture, and a UI/CLI split over one shared
core. What would move it from "solid, well-structured project" to "the project that gets
follow-up questions in an interview" — roughly in order of effort-to-impact:

1. **Deploy it and put the live link in the README/CV.** A project a reviewer can actually click
   and use in thirty seconds is worth more than any amount of description. Streamlit Community
   Cloud is free and this repo is already set up for it.
2. **Add LangSmith tracing.** A few lines of setup (`LANGSMITH_API_KEY` + `LANGSMITH_TRACING`
   env vars — LangChain picks these up automatically) gives you a shareable trace of exactly
   what each agent did, which tool calls it made, and what each one cost. This is the single
   highest-signal addition for demonstrating you understand *observability* in LLM systems, not
   just how to call `.invoke()`.
3. **Add a small evaluation set.** Five to ten fixed topics, run through the pipeline, scored by
   an LLM-as-judge rubric (or even just checking the Critic's own score stays above a threshold)
   as a CI-adjacent script. This is what separates "I built an agent" from "I know how to tell if
   an agent is actually working," which is the harder and more senior-sounding skill.
4. **Parallelize what's actually independent.** If you extend the Reader to scrape the top 2-3
   URLs instead of one (a natural, in-scope improvement — better sourcing, more defensible
   reports), fetch them concurrently (`asyncio.gather` or a thread pool) rather than serially.
   Small change, but "I identified and parallelized the independent part of a pipeline" is a
   concrete, specific thing to say about performance work.
5. **Cache search results per topic** (even a simple in-memory or SQLite cache keyed on the
   normalized topic string) to cut cost and latency on repeated/similar queries, and mention it
   explicitly as a cost-control decision — ties back to the "deployed under my own API key"
   constraint you already designed around.
6. **A short write-up** (a blog post, or even this file trimmed to the highlights) linked from
   your CV alongside the demo. Reviewers skim code for two minutes; a well-written explanation of
   *why* you made the SSRF-guard call, or *why* the pipeline degrades per-step, does more work
   per minute of their attention than the code itself.
7. **A GIF or short screen recording** in the README. Costs ten minutes, and it's the difference
   between a recruiter opening the repo and a recruiter scrolling past it.

Deliberately not on this list: adding auth, a database, or multi-provider support just to pad
scope. Each is a legitimate improvement but none of them changes what this project *demonstrates*
the way tracing, evaluation, or a live demo do — and a project that's easy to explain end-to-end
in five minutes is a stronger interview asset than a bigger one you have to summarize.
