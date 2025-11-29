"""API dependencies."""

from ..config import Settings, get_settings


def get_app_settings() -> Settings:
    """Get application settings dependency."""
    return get_settings()
