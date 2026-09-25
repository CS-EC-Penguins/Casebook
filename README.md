# Casebook

Casebook is a document Q&A system for a consulting firm's internal knowledge base. A user asks a question in plain English and receives an answer grounded in the firm's documents, with a citation to the source.

**The central rule:** answers come only from retrieved passages. If the information is not in the corpus, Casebook says so. It never generates answers from the model's general knowledge.

---

## What's in this repo

```
pipeline.py            Core RAG pipeline — split, index, retrieve, ask
agent.py               ReAct agent (Google Gen AI, no LangGraph)
graph_agent.py         LangGraph agent with structured citations
web_search_server.py   MCP server exposing Tavily web search to agents
agent_with_mcp.py      MCP smoke test (not a production agent)
evaluate.py            RAGAS evaluation harness for the pipeline
evaluate_agent.py      RAGAS evaluation harness for the graph agent
trajectory_judge.py    LLM judge for agent tool-routing behaviour
check_thresholds.py    CI gate — fails if RAGAS scores drop below threshold
test_graph_structured.py  Offline unit tests for graph agent structured output
test_trajectories.py   Live agent tool-routing assertions

new_corpus/            The four AI governance documents
evaluation/            Golden dataset, RAGAS config, and output files
```

---

## Prerequisites

### 1. Google Cloud access

You need access to the `pa-consulting-ai-eng` Google Cloud project with Vertex AI enabled. Authenticate with:

```bash
gcloud auth application-default login
```

### 2. Cloud SQL Auth Proxy

The vector database (pgvector on Cloud SQL) is not publicly accessible. You must run the proxy locally before any database operations:

```bash
cloud-sql-proxy pa-consulting-ai-eng:europe-west2:casebook-pg
```

Leave this running in a separate terminal while you work.

### 3. Environment variables

Source the provided file, or export the variables manually:

```bash
source env.sh
```

This sets:
- `GOOGLE_CLOUD_PROJECT` — GCP project ID
- `GOOGLE_CLOUD_LOCATION` — Vertex AI region (`europe-west4`)
- `PG_CONNECTION_STRING` — connection string for the pgvector database
- `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_BASE_URL` — observability tracing
- `PYTHONWARNINGS` — suppresses noisy deprecation warnings from the Google SDK

For web search (used by the agents), you also need:

```bash
export TAVILY_API_KEY="<your key>"
```

### 4. Python dependencies

Install with pip (a `requirements.txt` or `pyproject.toml` should list them — ask a teammate if one is missing):

Key packages: `langchain`, `langchain-google-vertexai`, `langchain-postgres`, `langgraph`, `langchain-mcp-adapters`, `google-genai`, `mcp`, `ragas`, `langfuse`, `tavily-python`, `tenacity`, `psycopg[binary]`.

---

## The corpus

The documents Casebook knows about live in `new_corpus/`:

| File | Content |
|---|---|
| `eu-ai-act.txt` | EU AI Act (Regulation 2024/1689) |
| `nist-ai-rmf.txt` | NIST AI Risk Management Framework 1.0 |
| `uk-aisi-report.txt` | UK AISI International AI Safety Report 2026 (text) |
| `international-ai-safety-report-2026.pdf` | International AI Safety Report 2026 (PDF, for reference) |
| `meridian-ai-governance-report.txt` | Meridian Centre AI Governance report |

The pipeline reads only `.txt` files via `load_documents()`.

---

## Quick start: ask a question

### Using the RAG pipeline

```bash
# First-time setup: embed the corpus and store it (takes a couple of minutes, costs money)
python pipeline.py index

# Then ask questions (cheap — no re-embedding)
python pipeline.py ask "What are the NIST AI RMF Core functions?"

# Ask with query rewriting (often improves retrieval)
python pipeline.py ask-rewritten "What does the EU law say about high-risk AI?"
```

### Using the LangGraph agent

The agent can also search the web for information outside the corpus:

```bash
python graph_agent.py "What are the main obligations for providers of high-risk AI systems?"
```

The agent returns structured JSON with numbered citations and a `status` field (`answered`, `insufficient_evidence`, or `iteration_limit`).

---

## How the RAG pipeline works

```
User question
     │
     ▼
rewrite_query()     ← optional; improves retrieval by reformulating the question
     │
     ▼
retrieve()          ← cosine similarity search against pgvector; returns top-4 passages
     │
     ▼
build_prompt()      ← numbers passages, adds system prompt
     │
     ▼
generate()          ← sends to Gemini 2.5 Flash; temperature=0 for determinism
     │
     ▼
answer + citations
```

All model calls are wrapped with `tenacity` retry logic (up to 5 attempts, exponential back-off) to handle Vertex AI quota errors gracefully.

---

## Pipeline commands reference

```bash
python pipeline.py split                     # chunk corpus, print first 4 chunks
python pipeline.py split 6                  # print chunk 6 and the 3 after it
python pipeline.py split-naive              # split with empty separator list (for comparison)
python pipeline.py index                    # embed all chunks and write to pgvector (run once)
python pipeline.py retrieve "question"      # show which passages would be retrieved
python pipeline.py prompt "question"        # print the full prompt sent to Gemini (no model call)
python pipeline.py ask "question"           # end-to-end answer
python pipeline.py ask-rewritten "question" # rewrite query first, then answer
```

> `index` clears the existing collection before re-embedding. Run it once; use `ask`/`retrieve` freely after.

---

## Tunable constants in pipeline.py

These are marked with `# DECISION` comments:

| Constant | Current value | Effect |
|---|---|---|
| `CHUNK_SIZE` | 800 | Max characters per chunk |
| `CHUNK_OVERLAP` | 150 | Characters shared between adjacent chunks |
| `TOP_K` | 4 | Number of passages retrieved per question |
| `SYSTEM_PROMPT` | (see file) | Controls grounding, citation format, and refusal to speculate |
| `SEPARATORS` | `["\n\n", "\n", ". ", " ", ""]` | Splitting order; paragraph breaks are preferred |

---

## Agents

Two agents extend the pipeline with tool use:

### `agent.py` — Direct ReAct agent
Uses the Google Gen AI SDK directly. The model/tool loop is visible in `run_agent()`. No LangGraph.

```bash
python agent.py "What does the NIST AI RMF say about governance?"
```

### `graph_agent.py` — LangGraph agent (recommended)
Uses LangGraph's `StateGraph` for the model → tools → model loop. Adds a second model pass to produce a structured response with numbered, verified citations.

```bash
python graph_agent.py "Compare how the EU AI Act and NIST AI RMF approach risk?"
```

Both agents have access to two tools:
- **`retrieve`** — searches the internal corpus
- **`web_search`** — searches the web via Tavily (for current events or out-of-corpus questions)

The web search tool is served by `web_search_server.py` over the MCP stdio protocol, and is discovered by the agents at runtime.

---

## Evaluation

Casebook uses [RAGAS](https://docs.ragas.io/) to measure answer quality against a human-written golden dataset.

```bash
# Evaluate the pipeline
python evaluate.py

# Evaluate the graph agent
python evaluate_agent.py

# Check scores against CI thresholds
python check_thresholds.py --results evaluation_results.json
```

### RAGAS metrics

| Metric | What it measures |
|---|---|
| `faithfulness` | Is the answer grounded in the retrieved passages? |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are the retrieved passages relevant? |
| `context_recall` | Did retrieval find all the needed information? |

The evaluation LLM is `gemini-2.5-pro`. Per-query results are saved to CSV files in `evaluation/`.

> **Do not edit `evaluation/golden_dataset.json`.** Reference answers are human-written. That is what makes the evaluation meaningful.

### Trajectory evaluation

For agents, `trajectory_judge.py` runs a separate dataset (`trajectory_dataset.json`) and asks an LLM to score whether the agent chose the right tools (0.0 = wrong tool, 1.0 = perfect):

```bash
python trajectory_judge.py
```

---

## Tests

```bash
# Fast offline tests — no model calls
python -m unittest test_graph_structured.py -v

# Live tests — make real agent calls (costs money)
pytest test_trajectories.py -v
```

---

## Observability

All major functions are decorated with `@observe()` from Langfuse. Traces appear at the URL configured in `LANGFUSE_BASE_URL`. This lets you inspect retrieved passages, prompts, and model responses for each query.

---

## Key conventions

- `retrieve(query, vector_store, k=4)` returns `[{"content": str, "source": str, "score": float}]`. This shape is a contract — both agents and the evaluation harness depend on it.
- Connection strings use the `postgresql+psycopg://` prefix (psycopg driver required by pgvector).
- `temperature=0` everywhere — answers change only when retrieval changes, not due to sampling variance.
- Never weaken the grounding constraint. Every answer-generation prompt must instruct the model to cite sources and refuse to speculate beyond the passages.
