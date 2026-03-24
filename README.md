# AI Dataset Analyzer

> Upload any CSV → LangChain agent runs statistical profiling, correlation detection, and semantic search → surfaces what actually matters.

```
repo/
├── backend/     FastAPI + LangChain + FAISS
└── frontend/    Static HTML/CSS/JS (GitHub Pages)
```

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI, Python 3.11+ |
| Agent | LangChain ZERO_SHOT_REACT |
| LLM | Gemini 1.5 Flash / Claude 3 Haiku / GPT-4o-mini |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB | FAISS (in-memory) |
| Frontend | Vanilla HTML/CSS/JS — zero dependencies |

## Setup in 3 steps

### 1. Backend

```bash
cd backend
pip install -r requirements.txt

export GOOGLE_API_KEY=your_key   # or ANTHROPIC_API_KEY / OPENAI_API_KEY
export LLM_PROVIDER=gemini       # gemini | claude | openai

uvicorn main:app --reload --port 8000
```

### 2. Frontend

Open `frontend/index.html` in a browser (or deploy to GitHub Pages).
Enter your backend URL in the input field — it's saved automatically.

### 3. Deploy backend (free)

**Railway** (recommended):
1. Push this repo to GitHub
2. New project → Deploy from GitHub → select `backend/` as root
3. Add env vars in the Railway dashboard
4. Done — Railway gives you a public HTTPS URL

**Render:**
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Root directory: `backend`

## What the agent analyzes

- Shape, column types, missing-value report
- Numeric summary (mean/std/min/max/quartiles)
- Categorical top-5 value distributions  
- Strong correlations — only |r| > 0.6 surfaces (signal, not noise)
- 3–5 numbered, number-backed insights



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

## License

MIT
