"""Database model types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Hardware:
    """Hardware device specification."""

    id: int
    name: str
    vendor: str
    chip: str
    memory_gb: float
    memory_bandwidth_gbs: float
    memory_type: str  # "unified" or "vram"
    gpu_cores: Optional[int] = None
    source: Optional[str] = None


@dataclass
class Model:
    """LLM model specification."""

    id: int
    family: str
    variant: Optional[str]
    size_b: float
    architecture: str = "dense"  # "dense" or "moe"
    context_default: int = 4096
    source: Optional[str] = None


@dataclass
class Quantization:
    """Model quantization."""

    id: int
    model_id: int
    quant_type: str
    size_gb: float
    bits_per_weight: float


@dataclass
class Benchmark:
    """Benchmark measurement."""

    id: int
    hardware_id: int
    quantization_id: int
    prompt_tokens: int
    generation_tokens: int
    generation_tps: float
    source: str
    context_length: Optional[int] = None
    backend: Optional[str] = None
    backend_version: Optional[str] = None
    prompt_tps: Optional[float] = None
    ttft_ms: Optional[float] = None
    measured_at: Optional[str] = None
