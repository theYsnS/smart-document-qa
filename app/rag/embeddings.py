"""Embedding generation and FAISS vector store management."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages HuggingFace embeddings and the FAISS vector store."""

    def __init__(
        self,
        model_name: str | None = None,
        store_path: Path | None = None,
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._store_path = store_path or settings.vector_store_path
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._vector_store: FAISS | None = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-initialize the embedding model."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
            )
        return self._embeddings

    @property
    def vector_store(self) -> FAISS:
        """Return the FAISS vector store, loading from disk or creating a new one."""
        if self._vector_store is None:
            self._vector_store = self._load_or_create_store()
        return self._vector_store

    def _load_or_create_store(self) -> FAISS:
        """Load existing FAISS index from disk or create an empty one."""
        index_path = self._store_path / "index.faiss"
        if index_path.exists():
            logger.info("Loading existing FAISS index from %s", self._store_path)
            return FAISS.load_local(
                str(self._store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        logger.info("Creating new FAISS index")
        embedding_dim = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_dim)
        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store and persist to disk.

        Args:
            documents: Chunked documents with metadata.

        Returns:
            List of document IDs assigned by the store.
        """
        if not documents:
            return []

        ids = self.vector_store.add_documents(documents)
        self._persist()
        logger.info("Added %d chunks to vector store", len(documents))
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents and return with similarity scores.

        Args:
            query: Search query text.
            k: Number of results to return.

        Returns:
            List of (document, score) tuples sorted by relevance.
            Lower scores indicate higher similarity (L2 distance).
        """
        top_k = k or settings.top_k
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks belonging to a specific document.

        Args:
            doc_id: The document identifier to remove.

        Returns:
            Number of chunks removed.
        """
        store = self.vector_store

        # Find all internal IDs matching this doc_id
        ids_to_delete: list[str] = []
        for store_id, doc in store.docstore._dict.items():
            if doc.metadata.get("doc_id") == doc_id:
                ids_to_delete.append(store_id)

        if ids_to_delete:
            store.delete(ids_to_delete)
            self._persist()
            logger.info("Deleted %d chunks for doc_id=%s", len(ids_to_delete), doc_id)

        return len(ids_to_delete)

    def get_all_doc_ids(self) -> set[str]:
        """Return all unique doc_ids currently in the store."""
        doc_ids: set[str] = set()
        for doc in self.vector_store.docstore._dict.values():
            did = doc.metadata.get("doc_id")
            if did:
                doc_ids.add(did)
        return doc_ids

    def get_chunks_by_doc_id(self, doc_id: str) -> list[Document]:
        """Return all chunks belonging to a document."""
        return [
            doc
            for doc in self.vector_store.docstore._dict.values()
            if doc.metadata.get("doc_id") == doc_id
        ]

    def _persist(self) -> None:
        """Save the FAISS index and docstore to disk."""
        self._store_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self._store_path))
        logger.info("Persisted vector store to %s", self._store_path)

    def reset(self) -> None:
        """Delete the vector store from disk and memory."""
        if self._store_path.exists():
            shutil.rmtree(self._store_path)
        self._vector_store = None
        logger.info("Vector store reset")
