"""
AI Dataset Analyzer - FastAPI Backend
Optimized: error handling, CORS, caching, async I/O, structured output
"""

import os
import uuid
import hashlib
import tempfile
import traceback
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.schema import Document

# ── LLM: swap freely between Gemini / Claude / OpenAI ──────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")   # "gemini" | "claude" | "openai"

def get_llm():
    if LLM_PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0,
                          openai_api_key=os.environ["OPENAI_API_KEY"])
    else:  # default: gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",          # gemini-pro is deprecated
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )

# ── Embeddings (cached so model loads only once) ────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # 5× faster than mpnet; same quality for retrieval
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ── Dataset analysis tool ────────────────────────────────────────────────────
def analyze_dataset(csv_path: str) -> str:
    """Return structured stats + inferred column types + missing-value report."""
    try:
        df = pd.read_csv(csv_path)

        # Basic shape
        n_rows, n_cols = df.shape
        dtypes = df.dtypes.astype(str).to_dict()

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / n_rows * 100).round(2)
        missing_report = {
            col: f"{missing[col]} ({missing_pct[col]}%)"
            for col in missing[missing > 0].index
        }

        # Numeric summary
        num_summary = df.describe(include="number").round(4).to_dict()

        # Categorical summary (top-5 values per column)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_summary = {
            col: df[col].value_counts().head(5).to_dict() for col in cat_cols
        }

        # Correlation (numeric only, warn if too many cols)
        corr_info = ""
        num_cols = df.select_dtypes(include="number").columns
        if 2 <= len(num_cols) <= 20:
            corr = df[num_cols].corr().round(3)
            # Surface only strong correlations (|r| > 0.6)
            strong = [
                f"{c1}↔{c2}: {corr.loc[c1, c2]}"
                for i, c1 in enumerate(num_cols)
                for c2 in num_cols[i + 1:]
                if abs(corr.loc[c1, c2]) > 0.6
            ]
            corr_info = strong if strong else ["No strong correlations found (|r|>0.6)"]

        report = {
            "shape": {"rows": n_rows, "columns": n_cols},
            "column_types": dtypes,
            "missing_values": missing_report or "None",
            "numeric_summary": num_summary,
            "categorical_summary": cat_summary,
            "strong_correlations": corr_info,
        }
        return str(report)
    except Exception as e:
        return f"Error analyzing dataset: {e}"


def semantic_search(query_and_path: str) -> str:
    """
    Accepts 'QUERY|||/path/to/file.csv' and runs semantic search
    over the CSV rows embedded into FAISS.
    """
    try:
        query, csv_path = query_and_path.split("|||", 1)
        loader = CSVLoader(csv_path.strip())
        docs = loader.load()
        vs = FAISS.from_documents(docs, get_embeddings())
        results = vs.similarity_search(query.strip(), k=5)
        return "\n---\n".join(d.page_content for d in results)
    except Exception as e:
        return f"Semantic search error: {e}"


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Dataset Analyzer",
    description="Upload a CSV, get an LLM-powered statistical + semantic analysis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your GitHub Pages domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": LLM_PROVIDER}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Read & cap file size (10 MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    # Write to temp file
    suffix = f"_{uuid.uuid4().hex[:8]}.csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        llm = get_llm()

        tools = [
            Tool(
                name="DatasetAnalyzer",
                func=analyze_dataset,
                description=(
                    "Analyzes a CSV file at the given path. "
                    "Returns shape, dtypes, missing values, numeric stats, "
                    "categorical distributions, and strong correlations. "
                    "Input: absolute file path string."
                ),
            ),
            Tool(
                name="SemanticSearch",
                func=semantic_search,
                description=(
                    "Semantically searches rows in a CSV. "
                    "Input format: 'your query|||/absolute/path/to/file.csv'"
                ),
            ),
        ]

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=6,
            handle_parsing_errors=True,
        )

        prompt = (
            f"You are a senior data analyst. Analyze the dataset at: {tmp_path}\n\n"
            "Steps:\n"
            "1. Use DatasetAnalyzer to get full statistics.\n"
            "2. Identify the most interesting patterns: outliers, skewed distributions, "
            "strong correlations, dominant categories.\n"
            "3. Highlight any data quality issues (missing values, suspicious dtypes).\n"
            "4. Give 3–5 concise, actionable insights a data scientist would care about.\n"
            "Be precise. Use numbers. No filler text."
        )

        result = agent.run(prompt)
        return {"status": "success", "analysis": result, "file": file.filename}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Always clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
