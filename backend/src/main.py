"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config import get_settings
from .db.connection import init_database
from .db.seed import seed_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - initialize resources on startup."""
    settings = get_settings()

    # Initialize and seed database
    init_database(settings.database_path)
    seed_database()

    yield

    # Cleanup on shutdown (if needed)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix=settings.api_prefix)

    # Health check at root level
    @app.get("/health")
    def health_check():
        return {"status": "ok", "version": settings.version}

    return app


app = create_app()
