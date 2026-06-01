from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "UCL Lab KAG Assistant"

    # LLM
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_chat_model: str = "nemotron-3-super-120b"
    openai_planner_model: str = ""  # lightweight model for planning/reasoning steps; falls back to chat_model if empty
    openai_embedding_model: str = "qwen3-embedding-8b"
    openai_embedding_dimensions: int = 4096

    # Google Drive
    google_service_account_json: str = "/app/secrets/ssl/google-service-account.json"
    google_drive_scopes: str = "https://www.googleapis.com/auth/drive.readonly"
    gdrive_folder_id: str = ""

    # KAG / OpenSPG
    kag_project_id: int = 1
    kag_namespace: str = "UCLLab"
    openspg_host: str = "http://openspg:8887"

    # Neo4j
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j@openspg"

    # Sync
    sync_interval_hours: float = 6.0

    # CORS
    cors_allow_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def google_drive_scope_list(self) -> list[str]:
        return [s.strip() for s in self.google_drive_scopes.split(",") if s.strip()]

    @property
    def cors_origins(self) -> list[str]:
        if self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [s.strip() for s in self.cors_allow_origins.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
