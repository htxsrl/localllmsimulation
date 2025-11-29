"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "LLM Performance Simulator"
    version: str = "0.1.0"
    debug: bool = False

    # Database path - use in-memory for production, file for development
    database_path: str = ":memory:"

    # Predictor model path
    predictor_path: str = str(Path(__file__).parent.parent / "data" / "predictor.pkl")

    # API settings
    api_prefix: str = "/api/v1"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
