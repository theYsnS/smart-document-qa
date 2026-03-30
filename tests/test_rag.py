"""Tests for RAG pipeline components."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.document_processor import (
    chunk_documents,
    clean_text,
    generate_doc_id,
    load_document,
    process_document,
)


class TestCleanText:
    def test_removes_control_characters(self):
        text = "Hello\x00World\x07Test"
        assert clean_text(text) == "Hello World Test"

    def test_normalizes_whitespace(self):
        text = "Hello   \t  World"
        assert clean_text(text) == "Hello World"

    def test_collapses_newlines(self):
        text = "Line1\n\n\n\n\nLine2"
        assert clean_text(text) == "Line1\n\nLine2"

    def test_strips_edges(self):
        text = "  \n  content  \n  "
        assert clean_text(text) == "content"


class TestLoadDocument:
    def test_load_txt(self, sample_txt_file: Path):
        docs = load_document(sample_txt_file)
        assert len(docs) == 1
        assert "artificial intelligence" in docs[0].page_content.lower()
        assert docs[0].metadata["source"] == "sample.txt"

    def test_load_nonexistent(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nonexistent.txt")

    def test_load_unsupported(self, tmp_path: Path):
        filepath = tmp_path / "data.csv"
        filepath.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(filepath)


class TestChunkDocuments:
    def test_chunking_produces_multiple_chunks(self, sample_large_txt: Path):
        docs = load_document(sample_large_txt)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self, sample_txt_file: Path):
        docs = load_document(sample_txt_file)
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_index_sequential(self, sample_large_txt: Path):
        docs = load_document(sample_large_txt)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestProcessDocument:
    def test_full_pipeline(self, sample_txt_file: Path):
        doc_id, chunks = process_document(sample_txt_file, chunk_size=100, chunk_overlap=20)
        assert isinstance(doc_id, str)
        assert len(doc_id) == 16
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["doc_id"] == doc_id

    def test_deterministic_doc_id(self, sample_txt_file: Path):
        id1, _ = process_document(sample_txt_file)
        id2, _ = process_document(sample_txt_file)
        assert id1 == id2


class TestGenerateDocId:
    def test_consistent_hash(self, sample_txt_file: Path):
        id1 = generate_doc_id(sample_txt_file)
        id2 = generate_doc_id(sample_txt_file)
        assert id1 == id2
        assert len(id1) == 16
