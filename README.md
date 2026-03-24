# Backend — AI Dataset Analyzer

FastAPI + LangChain agent that statistically and semantically analyzes any CSV.

## Quick Start

```bash
pip install -r requirements.txt
```

Set **one** of these environment variables depending on your LLM:

| Provider | Variable | LLM_PROVIDER value |
|----------|----------|-------------------|
| Google Gemini (default) | `GOOGLE_API_KEY` | `gemini` |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude` |
| OpenAI | `OPENAI_API_KEY` | `openai` |

```bash
# Example — Gemini (free tier)
export GOOGLE_API_KEY=your_key_here
export LLM_PROVIDER=gemini          # optional, gemini is default

uvicorn main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/analyze` | Upload CSV → AI analysis |

## Deploying (free options)

- **Railway** — connect GitHub repo, set env vars, done.
- **Render** — free tier works; set start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Fly.io** — `fly launch` in this directory.

## What the agent does

1. Runs `DatasetAnalyzer`: shape, dtypes, missing values, numeric stats, categorical top-5, strong correlations (|r| > 0.6)
2. Runs `SemanticSearch` if pattern exploration is needed
3. Returns 3–5 numbered, number-backed insights — no filler
