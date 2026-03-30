"""Tests for FastAPI endpoints."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_client():
    """Create a test client with fully mocked RAG components."""
    with patch("app.main.EmbeddingManager") as mock_em_cls, \
         patch("app.main.RAGChain") as mock_chain_cls:
        mock_em = MagicMock()
        mock_em.get_all_doc_ids.return_value = set()
        mock_em.add_documents.return_value = ["id1", "id2"]
        mock_em_cls.return_value = mock_em

        mock_chain = MagicMock()
        mock_chain_cls.return_value = mock_chain

        from app.main import app, document_registry
        document_registry.clear()

        with TestClient(app) as c:
            yield c, mock_em, mock_chain, document_registry


class TestHealthEndpoint:
    def test_health_check(self, api_client):
        client, *_ = api_client
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestDocumentUpload:
    def test_upload_txt_file(self, api_client):
        client, mock_em, _, registry = api_client
        content = b"This is a test document with some content."

        with patch("app.main.process_document") as mock_process:
            mock_process.return_value = ("abc123", [MagicMock(), MagicMock()])
            response = client.post(
                "/documents/upload",
                files={"file": ("test.txt", BytesIO(content), "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "abc123"
        assert data["filename"] == "test.txt"
        assert data["num_chunks"] == 2

    def test_upload_unsupported_format(self, api_client):
        client, *_ = api_client
        response = client.post(
            "/documents/upload",
            files={"file": ("test.csv", BytesIO(b"a,b,c"), "text/csv")},
        )
        assert response.status_code == 415

    def test_upload_duplicate_document(self, api_client):
        client, mock_em, _, registry = api_client

        with patch("app.main.process_document") as mock_process:
            mock_process.return_value = ("dup123", [MagicMock()])

            # First upload
            client.post(
                "/documents/upload",
                files={"file": ("doc.txt", BytesIO(b"content"), "text/plain")},
            )
            # Duplicate upload
            response = client.post(
                "/documents/upload",
                files={"file": ("doc.txt", BytesIO(b"content"), "text/plain")},
            )

        assert response.status_code == 400
        assert "already indexed" in response.json()["detail"]


class TestListDocuments:
    def test_list_empty(self, api_client):
        client, *_ = api_client
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["documents"] == []


class TestDeleteDocument:
    def test_delete_nonexistent(self, api_client):
        client, *_ = api_client
        response = client.delete("/documents/nonexistent")
        assert response.status_code == 404


class TestQueryEndpoint:
    def test_query_without_documents(self, api_client):
        client, *_ = api_client
        response = client.post(
            "/query",
            json={"question": "What is AI?"},
        )
        assert response.status_code == 400
        assert "No documents indexed" in response.json()["detail"]
