"""Database queries."""

from __future__ import annotations

from sqlite3 import Row
from typing import List, Optional, Tuple

from .connection import get_cursor
from .models import Benchmark, Hardware, Model, Quantization


def row_to_hardware(row: Row) -> Hardware:
    """Convert database row to Hardware object."""
    return Hardware(
        id=row["id"],
        name=row["name"],
        vendor=row["vendor"],
        chip=row["chip"],
        memory_gb=row["memory_gb"],
        memory_bandwidth_gbs=row["memory_bandwidth_gbs"],
        memory_type=row["memory_type"],
        gpu_cores=row["gpu_cores"],
        source=row["source"] if "source" in row.keys() else None,
    )


def row_to_model(row: Row) -> Model:
    """Convert database row to Model object."""
    return Model(
        id=row["id"],
        family=row["family"],
        variant=row["variant"],
        size_b=row["size_b"],
        architecture=row["architecture"],
        context_default=row["context_default"],
        source=row["source"] if "source" in row.keys() else None,
    )


def row_to_quantization(row: Row) -> Quantization:
    """Convert database row to Quantization object."""
    return Quantization(
        id=row["id"],
        model_id=row["model_id"],
        quant_type=row["quant_type"],
        size_gb=row["size_gb"],
        bits_per_weight=row["bits_per_weight"],
    )


# Hardware queries


def get_all_hardware(
    vendor: Optional[str] = None,
    min_memory_gb: Optional[float] = None,
    max_memory_gb: Optional[float] = None,
) -> List[Hardware]:
    """Get all hardware, optionally filtered."""
    query = "SELECT * FROM hardware WHERE 1=1"
    params: list = []

    if vendor:
        query += " AND vendor = ?"
        params.append(vendor)
    if min_memory_gb is not None:
        query += " AND memory_gb >= ?"
        params.append(min_memory_gb)
    if max_memory_gb is not None:
        query += " AND memory_gb <= ?"
        params.append(max_memory_gb)

    query += " ORDER BY vendor, memory_gb DESC"

    with get_cursor() as cursor:
        cursor.execute(query, params)
        return [row_to_hardware(row) for row in cursor.fetchall()]


def get_hardware_by_id(hardware_id: int) -> Optional[Hardware]:
    """Get hardware by ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM hardware WHERE id = ?", (hardware_id,))
        row = cursor.fetchone()
        return row_to_hardware(row) if row else None


def get_benchmark_count_for_hardware(hardware_id: int) -> int:
    """Get number of benchmarks for a hardware."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) as count FROM benchmarks WHERE hardware_id = ?",
            (hardware_id,),
        )
        row = cursor.fetchone()
        return row["count"] if row else 0


# Model queries


def get_all_models(
    family: Optional[str] = None,
    min_size_b: Optional[float] = None,
    max_size_b: Optional[float] = None,
) -> List[Model]:
    """Get all models, optionally filtered."""
    query = "SELECT * FROM models WHERE 1=1"
    params: list = []

    if family:
        query += " AND family = ?"
        params.append(family)
    if min_size_b is not None:
        query += " AND size_b >= ?"
        params.append(min_size_b)
    if max_size_b is not None:
        query += " AND size_b <= ?"
        params.append(max_size_b)

    query += " ORDER BY family, size_b"

    with get_cursor() as cursor:
        cursor.execute(query, params)
        return [row_to_model(row) for row in cursor.fetchall()]


def get_model_by_id(model_id: int) -> Optional[Model]:
    """Get model by ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        return row_to_model(row) if row else None


# Quantization queries


def get_quantizations_for_model(model_id: int) -> List[Quantization]:
    """Get all quantizations for a model."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM quantizations WHERE model_id = ? ORDER BY bits_per_weight DESC",
            (model_id,),
        )
        return [row_to_quantization(row) for row in cursor.fetchall()]


def get_quantization_by_id(quant_id: int) -> Optional[Quantization]:
    """Get quantization by ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM quantizations WHERE id = ?", (quant_id,))
        row = cursor.fetchone()
        return row_to_quantization(row) if row else None


# Benchmark queries


def get_benchmark_for_config(
    hardware_id: int,
    quantization_id: int,
    context_length: Optional[int] = None,
) -> Optional[Benchmark]:
    """Get closest benchmark for a hardware/quantization combination by context length."""
    with get_cursor() as cursor:
        if context_length:
            cursor.execute(
                """
                SELECT * FROM benchmarks
                WHERE hardware_id = ? AND quantization_id = ?
                ORDER BY ABS(COALESCE(context_length, prompt_tokens) - ?) LIMIT 1
                """,
                (hardware_id, quantization_id, context_length),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM benchmarks
                WHERE hardware_id = ? AND quantization_id = ?
                LIMIT 1
                """,
                (hardware_id, quantization_id),
            )

        row = cursor.fetchone()
        if not row:
            return None

        return Benchmark(
            id=row["id"],
            hardware_id=row["hardware_id"],
            quantization_id=row["quantization_id"],
            prompt_tokens=row["prompt_tokens"],
            generation_tokens=row["generation_tokens"],
            generation_tps=row["generation_tps"],
            source=row["source"],
            context_length=row["context_length"],
            backend=row["backend"],
            backend_version=row["backend_version"],
            prompt_tps=row["prompt_tps"],
            ttft_ms=row["ttft_ms"],
            measured_at=row["measured_at"],
        )


def get_nearest_benchmarks(
    target_bandwidth_gbs: float,
    target_memory_gb: float,
    target_model_size_gb: float,
    target_model_family: Optional[str] = None,
    limit: int = 4,
) -> List[Tuple[Benchmark, Hardware, str, str]]:
    """
    Find nearest benchmarks by hardware similarity (bandwidth, memory, model).

    Returns list of (benchmark, hardware, model_name, quant_type) tuples
    with similar hardware characteristics to provide meaningful comparisons.
    """
    with get_cursor() as cursor:
        # Get benchmarks with TTFT data
        cursor.execute(
            """
            SELECT
                b.*,
                h.id as h_id, h.name as h_name, h.vendor, h.chip,
                h.memory_gb, h.memory_bandwidth_gbs, h.memory_type, h.gpu_cores,
                m.family, m.size_b, q.quant_type, q.size_gb as quant_size_gb
            FROM benchmarks b
            JOIN hardware h ON b.hardware_id = h.id
            JOIN quantizations q ON b.quantization_id = q.id
            JOIN models m ON q.model_id = m.id
            WHERE b.ttft_ms IS NOT NULL AND b.generation_tps IS NOT NULL
            """
        )

        rows = cursor.fetchall()
        if not rows:
            return []

        # Calculate hardware similarity score for each benchmark
        benchmarks_with_score = []
        for row in rows:
            hw_bandwidth = row["memory_bandwidth_gbs"]
            hw_memory = row["memory_gb"]
            model_size = row["quant_size_gb"]
            model_family = row["family"]

            # Bandwidth similarity: ratio between 0.5 and 2.0 is good
            bw_ratio = hw_bandwidth / target_bandwidth_gbs if target_bandwidth_gbs > 0 else 1
            bw_score = abs(1 - bw_ratio)  # 0 = perfect match

            # Memory similarity
            mem_ratio = hw_memory / target_memory_gb if target_memory_gb > 0 else 1
            mem_score = abs(1 - mem_ratio)

            # Model size similarity
            size_ratio = model_size / target_model_size_gb if target_model_size_gb > 0 else 1
            size_score = abs(1 - size_ratio)

            # Model family bonus (prefer same family)
            family_bonus = 0 if target_model_family and model_family == target_model_family else 0.5

            # Combined similarity score (lower is better)
            # Weight: bandwidth most important, then model size, then memory
            similarity = (bw_score * 2.0) + (size_score * 1.5) + (mem_score * 1.0) + family_bonus

            benchmark = Benchmark(
                id=row["id"],
                hardware_id=row["hardware_id"],
                quantization_id=row["quantization_id"],
                prompt_tokens=row["prompt_tokens"],
                generation_tokens=row["generation_tokens"],
                generation_tps=row["generation_tps"],
                source=row["source"],
                context_length=row["context_length"],
                backend=row["backend"],
                backend_version=row["backend_version"],
                prompt_tps=row["prompt_tps"],
                ttft_ms=row["ttft_ms"],
                measured_at=row["measured_at"],
            )

            hardware = Hardware(
                id=row["h_id"],
                name=row["h_name"],
                vendor=row["vendor"],
                chip=row["chip"],
                memory_gb=row["memory_gb"],
                memory_bandwidth_gbs=row["memory_bandwidth_gbs"],
                memory_type=row["memory_type"],
                gpu_cores=row["gpu_cores"],
            )

            model_name = f"{row['family']} {row['size_b']}B"
            quant_type = row["quant_type"]

            benchmarks_with_score.append(
                (similarity, benchmark, hardware, model_name, quant_type)
            )

        # Sort by similarity (lower = more similar hardware)
        benchmarks_with_score.sort(key=lambda x: x[0])
        return [
            (b, h, m, q) for _, b, h, m, q in benchmarks_with_score[:limit]
        ]


def get_all_benchmarks() -> List[dict]:
    """Get all benchmarks with hardware and model info for the sources list."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                b.id,
                h.name as hardware_name,
                m.family as model_family,
                m.size_b as model_size_b,
                q.quant_type,
                COALESCE(b.context_length, b.prompt_tokens) as context_length,
                b.generation_tps,
                b.prompt_tps,
                b.ttft_ms,
                b.source
            FROM benchmarks b
            JOIN hardware h ON b.hardware_id = h.id
            JOIN quantizations q ON b.quantization_id = q.id
            JOIN models m ON q.model_id = m.id
            ORDER BY h.name, m.family, m.size_b
            """
        )
        return [
            {
                "id": row["id"],
                "hardware_name": row["hardware_name"],
                "model_family": row["model_family"],
                "model_size_b": row["model_size_b"],
                "quant_type": row["quant_type"],
                "context_length": row["context_length"],
                "generation_tps": row["generation_tps"],
                "prompt_tps": row["prompt_tps"],
                "ttft_ms": row["ttft_ms"],
                "source": row["source"],
            }
            for row in cursor.fetchall()
        ]


def get_similar_benchmark(
    quantization_id: int,
    memory_bandwidth_gbs: float,
) -> Optional[Tuple[Benchmark, Hardware]]:
    """Find a benchmark with similar hardware for the same model."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT b.*, h.*
            FROM benchmarks b
            JOIN hardware h ON b.hardware_id = h.id
            WHERE b.quantization_id = ?
            ORDER BY ABS(h.memory_bandwidth_gbs - ?)
            LIMIT 1
            """,
            (quantization_id, memory_bandwidth_gbs),
        )

        row = cursor.fetchone()
        if not row:
            return None

        benchmark = Benchmark(
            id=row["id"],
            hardware_id=row["hardware_id"],
            quantization_id=row["quantization_id"],
            prompt_tokens=row["prompt_tokens"],
            generation_tokens=row["generation_tokens"],
            generation_tps=row["generation_tps"],
            source=row["source"],
            context_length=row["context_length"],
            backend=row["backend"],
            backend_version=row["backend_version"],
            prompt_tps=row["prompt_tps"],
            ttft_ms=row["ttft_ms"],
            measured_at=row["measured_at"],
        )

        hardware = Hardware(
            id=row["hardware_id"],
            name=row["name"],
            vendor=row["vendor"],
            chip=row["chip"],
            memory_gb=row["memory_gb"],
            memory_bandwidth_gbs=row["memory_bandwidth_gbs"],
            memory_type=row["memory_type"],
            gpu_cores=row["gpu_cores"],
        )

        return benchmark, hardware
