"""Central configuration (pydantic-settings, values from .env).

Models are referenced as "<provider>:<model>" strings for LangChain
init_chat_model — never hard-code a model inside agent code (see ADR-0003).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers ---------------------------------------------------
    default_provider: str = "openai"
    openai_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Per-agent model map (override via env: MEDDX_AGENT_MODELS as JSON)
    agent_models: dict[str, str] = {
        "intake": "openai:gpt-4.1-mini",
        "hypothesis": "openai:gpt-4.1",
        "evidence": "openai:gpt-4.1",
        "devils_advocate": "openai:gpt-4.1",
        "root_cause": "openai:gpt-4.1",
        "synthesis": "openai:gpt-4.1",
    }

    # --- Storage ----------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "med_literature"

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "meddx"
    mysql_user: str = "meddx"
    mysql_password: str = "meddx"

    # --- Observability (Langfuse, self-hosted) ----------------------------
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- Ingestion ---------------------------------------------------------
    ncbi_tool_email: str = ""

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


settings = Settings()
