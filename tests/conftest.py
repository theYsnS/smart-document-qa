"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings


@pytest.fixture(autouse=True)
def _env_setup(tmp_path: Path) -> None:
    """Set environment variables for testing."""
    os.environ["OPENAI_API_KEY"] = "test-key-not-real"
    os.environ["VECTOR_STORE_PATH"] = str(tmp_path / "vector_store")
    os.environ["UPLOAD_DIR"] = str(tmp_path / "uploads")


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Create test settings with temporary paths."""
    return Settings(
        openai_api_key="test-key-not-real",
        vector_store_path=tmp_path / "vector_store",
        upload_dir=tmp_path / "uploads",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=50,
    )


@pytest.fixture()
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    filepath = tmp_path / "sample.txt"
    filepath.write_text(
        "Artificial intelligence (AI) is intelligence demonstrated by machines. "
        "Unlike natural intelligence displayed by animals including humans, "
        "AI uses algorithms and computational methods. "
        "Machine learning is a subset of AI that enables systems to learn "
        "from data without being explicitly programmed. "
        "Deep learning is a subset of machine learning based on artificial "
        "neural networks with representation learning.",
        encoding="utf-8",
    )
    return filepath


@pytest.fixture()
def sample_large_txt(tmp_path: Path) -> Path:
    """Create a larger text file to test chunking."""
    filepath = tmp_path / "large_sample.txt"
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Section {i + 1}: This is paragraph number {i + 1} of the document. "
            f"It contains information about topic {i + 1} which is relevant to "
            f"the overall subject matter. The details in this section cover "
            f"various aspects of the topic including methodology, results, "
            f"and conclusions drawn from the analysis of the data."
        )
    filepath.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return filepath


@pytest.fixture()
def mock_embeddings():
    """Mock HuggingFace embeddings to avoid downloading models in tests."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 384
    mock.embed_documents.return_value = [[0.1] * 384]
    return mock


@pytest.fixture()
def client():
    """Create a FastAPI test client with mocked dependencies."""
    with patch("app.main.EmbeddingManager") as mock_em_cls, \
         patch("app.main.RAGChain") as mock_chain_cls:
        mock_em = MagicMock()
        mock_em.get_all_doc_ids.return_value = set()
        mock_em_cls.return_value = mock_em

        mock_chain = MagicMock()
        mock_chain_cls.return_value = mock_chain

        from app.main import app

        with TestClient(app) as c:
            yield c
