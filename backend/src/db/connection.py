"""Database connection management."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

# Global in-memory connection - persists for app lifetime
_memory_connection: Optional[sqlite3.Connection] = None


def get_schema_sql() -> str:
    """Return the database schema SQL."""
    return """
    -- Hardware devices with specifications
    CREATE TABLE IF NOT EXISTS hardware (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        vendor TEXT NOT NULL,
        chip TEXT NOT NULL,
        memory_gb REAL NOT NULL,
        memory_bandwidth_gbs REAL NOT NULL,
        memory_type TEXT NOT NULL CHECK (memory_type IN ('unified', 'vram')),
        gpu_cores INTEGER,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- LLM models
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        family TEXT NOT NULL,
        variant TEXT,
        size_b REAL NOT NULL,
        architecture TEXT DEFAULT 'dense' CHECK (architecture IN ('dense', 'moe')),
        context_default INTEGER DEFAULT 4096,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(family, variant, size_b)
    );

    -- Quantizations for each model
    CREATE TABLE IF NOT EXISTS quantizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER NOT NULL REFERENCES models(id),
        quant_type TEXT NOT NULL,
        size_gb REAL NOT NULL,
        bits_per_weight REAL NOT NULL,
        UNIQUE(model_id, quant_type)
    );

    -- Real benchmark measurements
    CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hardware_id INTEGER NOT NULL REFERENCES hardware(id),
        quantization_id INTEGER NOT NULL REFERENCES quantizations(id),
        prompt_tokens INTEGER NOT NULL,
        generation_tokens INTEGER NOT NULL,
        context_length INTEGER,
        backend TEXT,
        backend_version TEXT,
        prompt_tps REAL,
        generation_tps REAL NOT NULL,
        ttft_ms REAL,
        source TEXT NOT NULL,
        measured_at TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_bench_hw ON benchmarks(hardware_id);
    CREATE INDEX IF NOT EXISTS idx_bench_quant ON benchmarks(quantization_id);
    CREATE INDEX IF NOT EXISTS idx_bench_tps ON benchmarks(generation_tps);
    CREATE INDEX IF NOT EXISTS idx_hw_mem ON hardware(memory_gb);
    CREATE INDEX IF NOT EXISTS idx_hw_bw ON hardware(memory_bandwidth_gbs);
    CREATE INDEX IF NOT EXISTS idx_quant_model ON quantizations(model_id);

    -- OCR tools for document parsing
    CREATE TABLE IF NOT EXISTS ocr_tools (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        vendor TEXT NOT NULL,
        tool_type TEXT NOT NULL CHECK (tool_type IN ('traditional', 'neural', 'hybrid')),
        requires_gpu INTEGER DEFAULT 0,
        min_vram_gb REAL,
        base_ram_gb REAL NOT NULL,
        description TEXT,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- OCR benchmarks
    CREATE TABLE IF NOT EXISTS ocr_benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hardware_id INTEGER NOT NULL REFERENCES hardware(id),
        tool_id INTEGER NOT NULL REFERENCES ocr_tools(id),
        pages_per_second REAL NOT NULL,
        ram_usage_gb REAL NOT NULL,
        vram_usage_gb REAL,
        cpu_cores_used INTEGER,
        is_gpu_accelerated INTEGER DEFAULT 0,
        document_type TEXT DEFAULT 'mixed' CHECK (document_type IN ('pdf_text', 'pdf_scanned', 'mixed')),
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_ocr_bench_hw ON ocr_benchmarks(hardware_id);
    CREATE INDEX IF NOT EXISTS idx_ocr_bench_tool ON ocr_benchmarks(tool_id);
    """


def init_database(db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Initialize the database with schema.

    For in-memory databases, returns a persistent connection.
    For file databases, creates the file and returns a new connection.
    """
    global _memory_connection

    if db_path == ":memory:":
        if _memory_connection is None:
            _memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            _memory_connection.row_factory = sqlite3.Row
            _memory_connection.executescript(get_schema_sql())
        return _memory_connection

    # File-based database
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(get_schema_sql())
    return conn


def get_connection() -> sqlite3.Connection:
    """Get the database connection."""
    global _memory_connection
    if _memory_connection is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _memory_connection


@contextmanager
def get_cursor() -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for database cursor."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
