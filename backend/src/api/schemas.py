"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# Hardware schemas


class HardwareBase(BaseModel):
    """Base hardware schema."""

    name: str
    vendor: str
    chip: str
    memory_gb: float
    memory_bandwidth_gbs: float
    memory_type: str  # "unified" or "vram"
    gpu_cores: Optional[int] = None


class HardwareResponse(HardwareBase):
    """Hardware response schema."""

    id: int


class HardwareDetailResponse(HardwareResponse):
    """Hardware detail with benchmark count."""

    benchmarks_count: int = 0


# Model schemas


class ModelBase(BaseModel):
    """Base model schema."""

    family: str
    variant: Optional[str] = None
    size_b: float
    architecture: str = "dense"  # "dense" or "moe"


class ModelResponse(ModelBase):
    """Model response schema."""

    id: int


class QuantizationResponse(BaseModel):
    """Quantization response schema."""

    id: int
    quant_type: str
    size_gb: float
    bits_per_weight: float


class ModelDetailResponse(ModelResponse):
    """Model detail with quantizations."""

    quantizations: List[QuantizationResponse] = []


class QuantizationDetailResponse(QuantizationResponse):
    """Quantization detail with model info."""

    model: ModelResponse


# Simulation schemas


class CustomHardwareInput(BaseModel):
    """Custom hardware specification input."""

    memory_gb: float = Field(gt=0)
    memory_bandwidth_gbs: float = Field(gt=0)
    memory_type: str = "unified"  # "unified" or "vram"
    gpu_cores: Optional[int] = None


class CustomModelInput(BaseModel):
    """Custom model specification input."""

    size_gb: float = Field(gt=0)
    bits_per_weight: float = Field(default=4.5, gt=0)
    is_moe: bool = False


class SimulationRequest(BaseModel):
    """Simulation request schema."""

    hardware_id: Optional[int] = None
    custom_hardware: Optional[CustomHardwareInput] = None
    quantization_id: Optional[int] = None
    custom_model: Optional[CustomModelInput] = None
    prompt_tokens: int = Field(default=8000, ge=1, le=1000000)


class SimilarBenchmark(BaseModel):
    """Similar benchmark info."""

    hardware_name: str
    generation_tps: float


class NearestBenchmark(BaseModel):
    """Nearest benchmark point for confidence visualization."""

    hardware_name: str
    model_name: str
    quant_type: str
    generation_tps: float
    ttft_ms: float
    source: Optional[str] = None


class SimulationResponse(BaseModel):
    """Simulation response schema."""

    can_run: bool
    memory_required_gb: float
    memory_available_gb: float
    generation_tps: float
    prompt_tps: Optional[float] = None
    ttft_ms: float
    confidence: float
    is_measured: bool
    similar_benchmark: Optional[SimilarBenchmark] = None
    nearest_benchmarks: List[NearestBenchmark] = []
    warnings: List[str] = []


# Search schemas


class MinRequirements(BaseModel):
    """Minimum hardware requirements."""

    memory_gb: float
    bandwidth_gbs: float


class HardwareInfo(BaseModel):
    """Hardware info for search results."""

    id: int
    name: str
    vendor: str
    chip: str
    memory_gb: float
    memory_bandwidth_gbs: float


class CompatibleHardware(BaseModel):
    """Compatible hardware result."""

    hardware: HardwareInfo
    estimated_tps: float
    meets_target: bool
    is_measured: bool
    confidence: float


class IncompatibleHardware(BaseModel):
    """Incompatible hardware result."""

    hardware: HardwareInfo
    reason: str  # "memory" or "bandwidth"
    shortfall_gb: Optional[float] = None


class HardwareSearchResponse(BaseModel):
    """Hardware search response."""

    min_requirements: MinRequirements
    compatible: List[CompatibleHardware]
    incompatible: List[IncompatibleHardware]


class ModelInfo(BaseModel):
    """Model info for search results."""

    id: int
    family: str
    variant: Optional[str] = None
    size_b: float


class QuantInfo(BaseModel):
    """Quantization info for search results."""

    id: int
    quant_type: str
    size_gb: float


class ModelSearchItem(BaseModel):
    """Model search result item."""

    model: ModelInfo
    quantization: QuantInfo
    estimated_tps: float
    can_run: bool
    confidence: float


class ModelSearchResponse(BaseModel):
    """Model search response."""

    models: List[ModelSearchItem]


# OCR schemas


class OcrToolResponse(BaseModel):
    """OCR tool response schema."""

    id: int
    name: str
    vendor: str
    tool_type: str  # 'traditional', 'neural', 'hybrid'
    requires_gpu: bool
    min_vram_gb: Optional[float] = None
    base_ram_gb: float
    description: Optional[str] = None


class OcrSimulationRequest(BaseModel):
    """OCR simulation request schema."""

    hardware_id: int
    tool_id: int
    page_count: int = Field(default=10, ge=1, le=10000)
    document_type: str = Field(default="mixed")  # 'pdf_text', 'pdf_scanned', 'mixed'


class OcrSimilarBenchmark(BaseModel):
    """Similar OCR benchmark info."""

    hardware_name: str
    pages_per_second: float


class OcrSimulationResponse(BaseModel):
    """OCR simulation response schema."""

    can_run: bool
    pages_per_second: float
    total_time_seconds: float
    ram_required_gb: float
    ram_available_gb: float
    vram_required_gb: Optional[float] = None
    vram_available_gb: Optional[float] = None
    is_gpu_accelerated: bool
    confidence: float
    is_measured: bool
    similar_benchmark: Optional[OcrSimilarBenchmark] = None
    warnings: List[str] = []


# ML Model info schemas


class FeatureImportance(BaseModel):
    """Feature importance item."""

    name: str
    importance: float  # Percentage (0-100)


class MLModelInfoResponse(BaseModel):
    """ML model information response."""

    algorithm: str
    r2_score: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    mean_percentage_error: Optional[float] = None
    training_samples: int
    feature_importance: List[FeatureImportance]
    use_log_transform: bool


# Benchmark list schema


class BenchmarkListItem(BaseModel):
    """Benchmark item for the sources list."""

    id: int
    hardware_name: str
    model_family: str
    model_size_b: float
    quant_type: str
    context_length: int
    generation_tps: float
    prompt_tps: Optional[float] = None
    ttft_ms: Optional[float] = None
    source: Optional[str] = None
