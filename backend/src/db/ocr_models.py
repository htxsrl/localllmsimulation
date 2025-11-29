"""OCR tool database models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OcrTool:
    """OCR tool/library."""

    id: int
    name: str
    vendor: str
    tool_type: str  # 'traditional', 'neural', 'hybrid'
    requires_gpu: bool
    min_vram_gb: Optional[float]
    base_ram_gb: float
    description: Optional[str]
    source: Optional[str]


@dataclass
class OcrBenchmark:
    """OCR performance benchmark."""

    id: int
    hardware_id: int
    tool_id: int
    pages_per_second: float
    ram_usage_gb: float
    vram_usage_gb: Optional[float]
    cpu_cores_used: Optional[int]
    is_gpu_accelerated: bool
    document_type: str  # 'pdf_text', 'pdf_scanned', 'mixed'
    source: Optional[str]
