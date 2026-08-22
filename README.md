# Multi-Agent Research System

A small multi-agent research assistant built on LangChain / LangGraph:

1. **Search Agent** — searches the web (Tavily) for recent, reliable information on a topic.
2. **Reader Agent** — picks the most relevant result and scrapes it for deeper content.
3. **Writer Chain** — drafts a structured research report from the gathered material.
4. **Critic Chain** — reviews and scores the report.

Available as a Streamlit UI (`app.py`) and a CLI (`main.py`), both built on the same
`run_research_pipeline` implementation in `src/pipeline.py`.

## Project layout

```
app.py              # Streamlit UI (thin entrypoint)
main.py             # CLI entrypoint
src/
  config.py          # secrets (st.secrets -> env -> .env), runtime limits, cheap-model allowlist
  prompts.py          # prompt templates
  agents.py            # LLM client factory + agent/chain builders
  tools.py              # Tavily web search + URL scraping (with an SSRF guard)
  pipeline.py            # search -> read -> write -> critique orchestration
  ui.py                   # theme CSS + sidebar model controls + results rendering
tests/               # offline unit tests (no API keys required)
```

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Running

CLI:

```bash
python main.py --topic "The impact of AI on the job market in 2026"
```

Streamlit UI:

```bash
streamlit run app.py
```

The sidebar lets you pick which model each of the four agents uses, from a fixed list of
cheap/low-cost models — see `src/config.py`. This allowlist is enforced both in the UI
and inside the pipeline itself, so it can't be bypassed by tampering with client-side state.

## Deploying to Render

The repo includes a [`render.yaml`](render.yaml) blueprint targeting Render's free tier.

1. Push the repo to GitHub, then in Render: **New → Blueprint**, pick the repo. Render reads
   `render.yaml` and configures the service automatically.
2. In the service's **Environment** tab, set `OPENAI_API_KEY` and `TAVILY_API_KEY`. They're
   declared with `sync: false` in the blueprint, which means Render prompts for the values
   instead of storing them in the repo — **never commit real keys**.
3. Deploy. The free tier spins the instance down after ~15 minutes idle, so the first request
   after a quiet period will be slow (cold start + container boot) — that's expected, not a
   broken deploy.

The app reads secrets via `st.secrets` first, falling back to environment variables /
`.env` for local development — see `src/config.py`. On Render only the environment-variable
path is used.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Tests are fully offline (no API keys required) — they cover the model allowlist, secret
resolution, the scraper's SSRF guard, and the pipeline's error handling.

## Cost & abuse guards

Because this is typically deployed with the owner's own API keys, a few guardrails are built
in: the model allowlist above, a per-session run limit, a max topic length, a bounded output
size for the writer, and request timeouts/retries on all outbound calls. See
`src/config.py` for the tunable limits.
