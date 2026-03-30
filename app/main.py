"""FastAPI application for RAG-based document Q&A."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.rag.chain import RAGChain
from app.rag.document_processor import SUPPORTED_EXTENSIONS, process_document
from app.rag.embeddings import EmbeddingManager
from app.schemas import (
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    WebSocketMessage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
embedding_manager: EmbeddingManager | None = None
rag_chain: RAGChain | None = None
document_registry: dict[str, DocumentInfo] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the vector store and RAG chain on startup."""
    global embedding_manager, rag_chain

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)

    embedding_manager = EmbeddingManager()
    rag_chain = RAGChain(embedding_manager)

    # Rebuild document registry from existing vector store
    for doc_id in embedding_manager.get_all_doc_ids():
        chunks = embedding_manager.get_chunks_by_doc_id(doc_id)
        if chunks:
            filename = chunks[0].metadata.get("source", "unknown")
            document_registry[doc_id] = DocumentInfo(
                doc_id=doc_id,
                filename=filename,
                num_chunks=len(chunks),
                uploaded_at=datetime.now(timezone.utc),
                file_size_bytes=0,
            )

    logger.info("Application started — %d documents in index", len(document_registry))
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="Smart Document Q&A",
    description="RAG-based document Q&A system with LangChain, FAISS, and FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    responses={400: {"model": ErrorResponse}, 415: {"model": ErrorResponse}},
    tags=["Documents"],
    summary="Upload and index a document",
)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
    """Upload a PDF, DOCX, or TXT file and index it for Q&A."""
    assert embedding_manager is not None

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Save uploaded file
    file_id = uuid.uuid4().hex[:12]
    save_path = settings.upload_dir / f"{file_id}_{file.filename}"
    content = await file.read()
    save_path.write_bytes(content)

    try:
        doc_id, chunks = process_document(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to process document: {e}")

    # Check for duplicate
    if doc_id in document_registry:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Document already indexed (doc_id={doc_id})",
        )

    embedding_manager.add_documents(chunks)

    document_registry[doc_id] = DocumentInfo(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
        uploaded_at=datetime.now(timezone.utc),
        file_size_bytes=len(content),
    )

    logger.info("Indexed document %s (%d chunks)", file.filename, len(chunks))
    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Q&A"],
    summary="Ask a question about uploaded documents",
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Ask a natural-language question and get a grounded answer with sources."""
    assert rag_chain is not None

    if not document_registry:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a document first.",
        )

    result = await rag_chain.aquery(
        question=request.question,
        session_id=request.session_id,
        top_k=request.top_k,
    )

    sources = [
        SourceDocument(
            content=doc.page_content[:500],
            document_name=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=0.0,
        )
        for doc in result["source_documents"]
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        confidence=result["confidence"],
        session_id=request.session_id,
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List all indexed documents",
)
async def list_documents() -> DocumentListResponse:
    """Return a list of all indexed documents with metadata."""
    docs = list(document_registry.values())
    return DocumentListResponse(documents=docs, total=len(docs))


@app.delete(
    "/documents/{doc_id}",
    tags=["Documents"],
    summary="Remove an indexed document",
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(doc_id: str) -> dict[str, str]:
    """Delete a document and all its chunks from the vector store."""
    assert embedding_manager is not None

    if doc_id not in document_registry:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    removed = embedding_manager.delete_by_doc_id(doc_id)
    del document_registry[doc_id]

    logger.info("Deleted document %s (%d chunks removed)", doc_id, removed)
    return {"message": f"Document {doc_id} deleted ({removed} chunks removed)"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming Q&A chat."""
    assert rag_chain is not None

    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = WebSocketMessage.model_validate(json.loads(raw))
            except Exception:
                await websocket.send_json({"error": "Invalid message format"})
                continue

            if not document_registry:
                await websocket.send_json(
                    {"error": "No documents indexed. Upload a document first."}
                )
                continue

            try:
                async for token in rag_chain.astream(
                    question=msg.question,
                    session_id=msg.session_id,
                    top_k=msg.top_k,
                ):
                    await websocket.send_json({"token": token})

                await websocket.send_json({"done": True})
            except Exception as e:
                logger.exception("Error during streaming")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
