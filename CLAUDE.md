# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UCL Lab KAG Assistant — a Knowledge-Augmented Generation (KAG) system that answers questions about UCL lab resources by combining Neo4j knowledge-graph reasoning with vector semantic search. Documents are ingested from Google Drive and indexed via the OpenSPG-KAG SDK.

## Running the Stack

```bash
# Full stack (all services + app)
docker compose up --build

# Rebuild only the FastAPI app after code changes
docker compose build app && docker compose up app

# View app logs
docker compose logs -f app

# Check if the app is ready
curl http://localhost:8001/api/health
curl http://localhost:8001/api/status

# Send a chat query
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who manages the GPU cluster?"}'
```

The app is served on **port 8001** (mapped to container port 8000). The frontend is served as static files from `./frontend/` — no build step needed.

## Service Architecture

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `mysql` | openspg-mysql | 3306 | Metadata store for OpenSPG control plane |
| `neo4j` | openspg-neo4j | 7474, 7687 | Knowledge graph + vector index |
| `minio` | openspg-minio | 9000, 9001 | Object storage for KAG artefacts |
| `openspg` | openspg-server | 8887 | Java-based OpenSPG control plane (project/schema API) |
| `app` | Built locally | 8001 | FastAPI application |

**Startup order:** MySQL → Neo4j + MinIO → OpenSPG (waits ~120 s) → app.

**OpenSPG is the control plane; Neo4j holds all actual graph data.** The `openspg` container must be healthy before the app can initialise.

## Backend Architecture (`backend/`)

```
main.py          — FastAPI app, 3 routes: GET /api/health, GET /api/status, POST /api/chat
config.py        — Pydantic Settings; all config from .env
schemas.py       — ChatRequest, ChatResponse, GraphFact, StatusResponse
services/
  kag_service.py    — Wraps KAG builder (NER→triples→Neo4j) and solver (hybrid retrieval→LLM)
  gdrive_loader.py  — Google Drive API client (list + download)
  sync_service.py   — ingest_drive_folder() and periodic_sync() background tasks
  document_parser.py — parse_bytes() for PDF/DOCX/PPTX/XLSX/TXT/MD/CSV
entrypoint.sh    — envsubst into kag_config.yaml.template, then uvicorn --reload
```

**Key data flow:**

1. On startup, `KAGService.initialize()` uses `ProjectClient.create()` (Python API) to create/find the OpenSPG project, then runs `knext schema commit` via subprocess to push the schema from `schema/UCLLab.schema`.
2. Background task `ingest_drive_folder()` downloads each Drive file → parses text → `KAGService.build_document()` writes entities/triples to Neo4j via `DefaultUnstructuredBuilderChain` (TXTReader → LengthSplitter → SchemaConstraintExtractor + UclLabNERPrompt → BatchVectorizer → KGWriter). Progress is checkpointed in `/app/ckpt/KGWriter/cache.db`.
3. `KAGService.solve()` uses `SolverMain().ainvoke()` with `usePipeline: "default_pipeline"` (naive RAG, `score_threshold=0.65`). The solver queries `UCLLab.Chunk` nodes via OpenSPG vector search and generates answers with the LLM.
4. Periodic sync runs every `SYNC_INTERVAL_HOURS` (default 6); files are deduped by MD5.

**Neo4j database:** Graph data is in the `ucllab` database (not the default `neo4j` db). Query it with: `cypher-shell --database ucllab`.

**KGWriter checkpoint:** If Neo4j is wiped (e.g., after `docker compose down -v`), delete `/app/ckpt/KGWriter/cache.db` inside the container before re-ingesting, otherwise KGWriter will skip all records thinking they're already written.

**Solver pipeline:** Uses `default_pipeline` (naive_rag.yaml) — vector chunk retrieval with `score_threshold=0.65`. The `think_pipeline` (deep_thought.yaml) is available for complex multi-hop reasoning but requires `chunk_index` with proper index_list configuration.

## Configuration

Copy `.env.example` → `.env` and fill in:

```env
OPENAI_BASE_URL=https://b225.54ucl.com/capystar/v1   # custom OpenAI-compatible endpoint
OPENAI_API_KEY=<key>
OPENAI_CHAT_MODEL=nemotron-3-super-120b
OPENAI_EMBEDDING_MODEL=qwen3-embedding-8b
OPENAI_EMBEDDING_DIMENSIONS=4096

GDRIVE_FOLDER_ID=<Google Drive folder ID to ingest>
GOOGLE_SERVICE_ACCOUNT_JSON=/app/secrets/ssl/google-service-account.json

KAG_PROJECT_ID=1
KAG_NAMESPACE=UCLLab
# OPENSPG_HOST is injected by docker-compose; do not set manually when using compose

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j@openspg

SYNC_INTERVAL_HOURS=6
CORS_ALLOW_ORIGINS=*
```

`kag_config.yaml` is a template — `entrypoint.sh` runs `envsubst` on it at container start to produce the live config. Never edit the generated `/app/kag_config.yaml` inside the container directly.

## Knowledge Graph Schema

Defined in `schema/UCLLab.schema` (OpenSPG DSL). 11 entity types: **Person**, **Project**, **Task**, **MeetingRecord**, **Decision**, **ResearchPlan**, **Patent**, **ComputeResource**, **Equipment**, **Course**, **ExternalParty**, **Competition**, plus **Chunk** (text+vector). The builder uses `SchemaConstraintExtractor` with `UclLabNERPrompt` (Chinese, meeting-note example) — LLM is constrained to only use defined schema categories. **Critical:** the schema file must be named `UCLLab.schema` (matching the namespace) for `knext schema commit` to find it.

## Operational Notes

- The `git` package is explicitly installed in the Dockerfile (needed by the KAG SDK's `gitpython` dependency).
- If Docker Desktop's metadata DB (`meta.db`) gets corrupted (shows `input/output error` on build): Troubleshoot → Restart, or Troubleshoot → Clean/Purge data.
- After a full volume wipe (`docker compose down -v`), delete `/app/ckpt/` checkpoints or KGWriter will skip re-ingestion.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
