# Smart Document Q&A

[![CI](https://github.com/theYsnS/smart-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/theYsnS/smart-document-qa/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An intelligent document Q&A system powered by **Retrieval-Augmented Generation (RAG)**. Upload documents in multiple formats, and ask natural-language questions — the system retrieves relevant passages and generates grounded answers with source citations.

## Architecture

```
                         Smart Document Q&A — RAG Pipeline
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                                                                         │
 │   ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌───────────────┐   │
 │   │ Document  │───▶│ Chunking  │───▶│ Embedding │───▶│ FAISS Vector  │   │
 │   │ Upload    │    │ (Recursive│    │ (sentence-│    │ Store         │   │
 │   │ PDF/DOCX/ │    │  Splitter)│    │ transform.)│   │               │   │
 │   │ TXT       │    └───────────┘    └───────────┘    └───────┬───────┘   │
 │   └──────────┘                                               │           │
 │                                                               │           │
 │   ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌───────▼───────┐   │
 │   │ Response  │◀───│ LLM       │◀───│ Prompt +  │◀───│ Similarity    │   │
 │   │ + Sources │    │ Generation│    │ Context   │    │ Search (Top-K)│   │
 │   └──────────┘    └───────────┘    └───────────┘    └───────────────┘   │
 │                                                                         │
 │   ┌─────────────────────────────────────────────────────────────────┐   │
 │   │              Conversation Memory (Sliding Window)               │   │
 │   └─────────────────────────────────────────────────────────────────┘   │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-format document ingestion** — PDF, DOCX, and plain text files
- **Intelligent chunking** — Recursive character splitting with configurable overlap for context preservation
- **Semantic search** — HuggingFace sentence-transformer embeddings with FAISS similarity search
- **Grounded answers** — LLM responses cite source documents and passages
- **Conversation memory** — Sliding-window buffer retains multi-turn context
- **Streaming responses** — Real-time token streaming via WebSocket
- **Confidence scoring** — Each answer includes a retrieval confidence score
- **REST + WebSocket API** — Full CRUD for documents, query endpoint, and streaming chat
- **Production-ready** — Dockerized, tested, CI/CD pipeline included

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI, Uvicorn |
| RAG Orchestration | LangChain |
| Embeddings | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (CPU) |
| LLM | OpenAI GPT-3.5 / GPT-4 (configurable) |
| Document Parsing | PyPDF, python-docx |
| Testing | pytest, httpx |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Linting | Ruff |

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key (or compatible endpoint)

### Installation

```bash
git clone https://github.com/theYsnS/smart-document-qa.git
cd smart-document-qa

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Reference

### Upload a Document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf"
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5}'
```

### List Documents

```bash
curl http://localhost:8000/documents
```

### Delete a Document

```bash
curl -X DELETE http://localhost:8000/documents/{doc_id}
```

### Streaming Chat (WebSocket)

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/chat");
ws.send(JSON.stringify({ question: "Summarize the report" }));
ws.onmessage = (e) => console.log(e.data);
```

## Configuration

All settings are configurable via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `LLM_MODEL` | `gpt-3.5-turbo` | LLM model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `MEMORY_WINDOW` | `5` | Conversation memory window size |
| `VECTOR_STORE_PATH` | `./vector_store` | FAISS index persistence path |

## Testing

```bash
pytest -v
```

## Project Structure

```
smart-document-qa/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   ├── config.py             # Settings and configuration
│   ├── schemas.py            # Pydantic request/response models
│   └── rag/
│       ├── __init__.py
│       ├── document_processor.py   # Document loading and chunking
│       ├── embeddings.py           # Embedding and vector store management
│       └── chain.py                # RAG chain and conversation memory
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   └── test_rag.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
