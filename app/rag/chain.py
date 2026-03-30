"""RAG chain construction with conversation memory and source tracking."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import AsyncIterator

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

QA_PROMPT_TEMPLATE = """\
You are a knowledgeable assistant that answers questions based strictly on the \
provided context. If the context does not contain enough information to answer \
the question, say so clearly — do not fabricate information.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Provide a clear, well-structured answer. Reference specific parts of the context \
when possible. If the answer spans multiple topics, use bullet points or numbered lists.\
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=QA_PROMPT_TEMPLATE,
)

CONDENSE_QUESTION_TEMPLATE = """\
Given the following conversation and a follow-up question, rephrase the \
follow-up question to be a standalone question that captures all necessary \
context from the conversation history.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:\
"""

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=CONDENSE_QUESTION_TEMPLATE,
)


class RAGChain:
    """Manages the conversational RAG chain with session-based memory."""

    def __init__(self, embedding_manager: EmbeddingManager) -> None:
        self._embedding_manager = embedding_manager
        self._memories: dict[str, ConversationBufferWindowMemory] = defaultdict(
            self._create_memory
        )
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-initialize the LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_base_url,
                streaming=True,
            )
        return self._llm

    @staticmethod
    def _create_memory() -> ConversationBufferWindowMemory:
        """Create a new conversation memory instance."""
        return ConversationBufferWindowMemory(
            k=settings.memory_window,
            memory_key="chat_history",
            return_messages=False,
            output_key="answer",
        )

    def _build_chain(
        self,
        session_id: str,
    ) -> ConversationalRetrievalChain:
        """Build a ConversationalRetrievalChain for the given session."""
        retriever = self._embedding_manager.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.top_k},
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self._memories[session_id],
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            condense_question_prompt=CONDENSE_PROMPT,
            return_source_documents=True,
            verbose=False,
        )
        return chain

    def query(
        self,
        question: str,
        session_id: str = "default",
        top_k: int | None = None,
    ) -> dict:
        """Run a synchronous RAG query.

        Args:
            question: The user's question.
            session_id: Conversation session identifier.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with keys: answer, source_documents, confidence.
        """
        if top_k and top_k != settings.top_k:
            retriever = self._embedding_manager.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
        else:
            retriever = None

        chain = self._build_chain(session_id)
        if retriever is not None:
            chain.retriever = retriever

        result = chain.invoke({"question": question})

        source_docs: list[Document] = result.get("source_documents", [])
        confidence = self._compute_confidence(question, source_docs)

        return {
            "answer": result["answer"],
            "source_documents": source_docs,
            "confidence": confidence,
        }

    async def aquery(
        self,
        question: str,
        session_id: str = "default",
        top_k: int | None = None,
    ) -> dict:
        """Run an async RAG query.

        Args:
            question: The user's question.
            session_id: Conversation session identifier.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with keys: answer, source_documents, confidence.
        """
        chain = self._build_chain(session_id)

        if top_k and top_k != settings.top_k:
            chain.retriever = self._embedding_manager.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )

        result = await chain.ainvoke({"question": question})

        source_docs: list[Document] = result.get("source_documents", [])
        confidence = self._compute_confidence(question, source_docs)

        return {
            "answer": result["answer"],
            "source_documents": source_docs,
            "confidence": confidence,
        }

    async def astream(
        self,
        question: str,
        session_id: str = "default",
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream the RAG response token by token.

        Retrieves context first, then streams the LLM generation.
        """
        k = top_k or settings.top_k
        results = self._embedding_manager.similarity_search_with_score(question, k=k)

        context_parts: list[str] = []
        for doc, _score in results:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            header = f"[Source: {source}" + (f", Page {page}]" if page else "]")
            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        memory = self._memories[session_id]
        chat_history = memory.buffer if hasattr(memory, "buffer") else ""

        prompt = QA_PROMPT.format(
            context=context,
            chat_history=chat_history,
            question=question,
        )

        full_response: list[str] = []
        async for chunk in self.llm.astream(prompt):
            token = chunk.content
            if token:
                full_response.append(token)
                yield token

        # Update memory with the full exchange
        full_answer = "".join(full_response)
        memory.save_context(
            {"question": question},
            {"answer": full_answer},
        )

    def _compute_confidence(
        self,
        query: str,
        source_docs: list[Document],
    ) -> float:
        """Compute a retrieval confidence score based on similarity distances.

        Uses the FAISS L2 distances: lower distance means higher confidence.
        Returns a value between 0.0 and 1.0.
        """
        if not source_docs:
            return 0.0

        results = self._embedding_manager.similarity_search_with_score(
            query, k=len(source_docs)
        )
        if not results:
            return 0.0

        scores = [score for _, score in results]
        avg_score = sum(scores) / len(scores)

        # Convert L2 distance to a 0-1 confidence using exponential decay
        # Typical L2 distances for normalized embeddings range from 0 to 2
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + avg_score)))
        return round(confidence, 4)

    def clear_session(self, session_id: str) -> None:
        """Clear conversation memory for a specific session."""
        if session_id in self._memories:
            self._memories[session_id].clear()
            del self._memories[session_id]
