from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # OpenAI / LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.2

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5

    # Conversation memory
    memory_window: int = 5

    # Storage
    vector_store_path: Path = Path("./vector_store")
    upload_dir: Path = Path("./uploads")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
