"""OCR-related database queries."""

from __future__ import annotations

from typing import List, Optional

from .connection import get_cursor
from .ocr_models import OcrBenchmark, OcrTool


def get_all_ocr_tools() -> List[OcrTool]:
    """Get all OCR tools."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM ocr_tools ORDER BY name")
        return [
            OcrTool(
                id=row["id"],
                name=row["name"],
                vendor=row["vendor"],
                tool_type=row["tool_type"],
                requires_gpu=bool(row["requires_gpu"]),
                min_vram_gb=row["min_vram_gb"],
                base_ram_gb=row["base_ram_gb"],
                description=row["description"],
                source=row["source"],
            )
            for row in cursor.fetchall()
        ]


def get_ocr_tool_by_id(tool_id: int) -> Optional[OcrTool]:
    """Get OCR tool by ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM ocr_tools WHERE id = ?", (tool_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return OcrTool(
            id=row["id"],
            name=row["name"],
            vendor=row["vendor"],
            tool_type=row["tool_type"],
            requires_gpu=bool(row["requires_gpu"]),
            min_vram_gb=row["min_vram_gb"],
            base_ram_gb=row["base_ram_gb"],
            description=row["description"],
            source=row["source"],
        )


def get_ocr_benchmark(
    hardware_id: int,
    tool_id: int,
    document_type: str = "mixed",
) -> Optional[OcrBenchmark]:
    """Get OCR benchmark for hardware/tool combination."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT * FROM ocr_benchmarks
            WHERE hardware_id = ? AND tool_id = ?
            ORDER BY ABS(
                CASE document_type
                    WHEN ? THEN 0
                    WHEN 'mixed' THEN 1
                    ELSE 2
                END
            )
            LIMIT 1
            """,
            (hardware_id, tool_id, document_type),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return OcrBenchmark(
            id=row["id"],
            hardware_id=row["hardware_id"],
            tool_id=row["tool_id"],
            pages_per_second=row["pages_per_second"],
            ram_usage_gb=row["ram_usage_gb"],
            vram_usage_gb=row["vram_usage_gb"],
            cpu_cores_used=row["cpu_cores_used"],
            is_gpu_accelerated=bool(row["is_gpu_accelerated"]),
            document_type=row["document_type"],
            source=row["source"],
        )


def get_similar_ocr_benchmark(
    tool_id: int,
    memory_bandwidth_gbs: float,
) -> Optional[tuple]:
    """Find a similar OCR benchmark by hardware bandwidth."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT ob.*, h.name as hw_name, h.memory_bandwidth_gbs
            FROM ocr_benchmarks ob
            JOIN hardware h ON ob.hardware_id = h.id
            WHERE ob.tool_id = ?
            ORDER BY ABS(h.memory_bandwidth_gbs - ?)
            LIMIT 1
            """,
            (tool_id, memory_bandwidth_gbs),
        )
        row = cursor.fetchone()
        if not row:
            return None
        benchmark = OcrBenchmark(
            id=row["id"],
            hardware_id=row["hardware_id"],
            tool_id=row["tool_id"],
            pages_per_second=row["pages_per_second"],
            ram_usage_gb=row["ram_usage_gb"],
            vram_usage_gb=row["vram_usage_gb"],
            cpu_cores_used=row["cpu_cores_used"],
            is_gpu_accelerated=bool(row["is_gpu_accelerated"]),
            document_type=row["document_type"],
            source=row["source"],
        )
        return benchmark, row["hw_name"], row["memory_bandwidth_gbs"]
