from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for document Q&A queries."""

    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    session_id: str = Field(default="default", description="Session ID for conversation memory")


class SourceDocument(BaseModel):
    """A source passage used to generate the answer."""

    content: str = Field(..., description="Text content of the chunk")
    document_name: str = Field(..., description="Original document filename")
    page: int | None = Field(default=None, description="Page number if available")
    score: float = Field(..., description="Similarity score (lower is more similar)")


class QueryResponse(BaseModel):
    """Response body for document Q&A queries."""

    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] = Field(default_factory=list, description="Source passages")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Retrieval confidence score"
    )
    session_id: str = Field(..., description="Session ID used")


class DocumentUploadResponse(BaseModel):
    """Response after uploading and indexing a document."""

    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    num_chunks: int = Field(..., description="Number of chunks created")
    message: str = Field(default="Document indexed successfully")


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    doc_id: str
    filename: str
    num_chunks: int
    uploaded_at: datetime
    file_size_bytes: int


class DocumentListResponse(BaseModel):
    """Response listing all indexed documents."""

    documents: list[DocumentInfo]
    total: int


class WebSocketMessage(BaseModel):
    """Message format for WebSocket chat."""

    question: str = Field(..., min_length=1)
    session_id: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
