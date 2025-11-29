"""OCR simulation service for document parsing performance estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..db import queries
from ..db import ocr_queries


@dataclass
class OcrSimulationResult:
    """Result of an OCR performance simulation."""

    can_run: bool
    pages_per_second: float
    total_time_seconds: float
    ram_required_gb: float
    ram_available_gb: float
    vram_required_gb: Optional[float]
    vram_available_gb: Optional[float]
    is_gpu_accelerated: bool
    confidence: float
    is_measured: bool
    similar_benchmark: Optional[dict]
    warnings: List[str]


def simulate_ocr(
    hardware_id: int,
    tool_id: int,
    page_count: int,
    document_type: str = "mixed",
) -> OcrSimulationResult:
    """
    Simulate OCR/document parsing performance.

    Args:
        hardware_id: ID of hardware to simulate
        tool_id: ID of OCR tool
        page_count: Number of pages to process
        document_type: 'pdf_text', 'pdf_scanned', or 'mixed'

    Returns:
        OcrSimulationResult with performance estimates
    """
    warnings: list[str] = []

    # Get hardware specs
    hw = queries.get_hardware_by_id(hardware_id)
    if not hw:
        raise ValueError(f"Hardware not found: {hardware_id}")

    # Get tool specs
    tool = ocr_queries.get_ocr_tool_by_id(tool_id)
    if not tool:
        raise ValueError(f"OCR tool not found: {tool_id}")

    # Check VRAM requirements for GPU tools
    vram_available = None
    vram_required = None
    is_gpu_accelerated = False

    if tool.requires_gpu:
        if hw.memory_type == "unified":
            # Apple Silicon - use unified memory
            vram_available = hw.memory_gb
            vram_required = tool.min_vram_gb
            is_gpu_accelerated = True
        elif hw.memory_type == "vram":
            vram_available = hw.memory_gb
            vram_required = tool.min_vram_gb
            is_gpu_accelerated = True
        else:
            warnings.append(f"{tool.name} works best with GPU acceleration")

    # Check if tool can run
    ram_required = tool.base_ram_gb
    can_run = True

    if vram_required and vram_available:
        if vram_required > vram_available:
            can_run = False
            warnings.append(
                f"Insufficient VRAM: {tool.name} needs {vram_required}GB, "
                f"hardware has {vram_available}GB"
            )

    if ram_required > hw.memory_gb:
        can_run = False
        warnings.append(
            f"Insufficient RAM: {tool.name} needs {ram_required}GB, "
            f"hardware has {hw.memory_gb}GB"
        )

    if not can_run:
        return OcrSimulationResult(
            can_run=False,
            pages_per_second=0,
            total_time_seconds=0,
            ram_required_gb=ram_required,
            ram_available_gb=hw.memory_gb,
            vram_required_gb=vram_required,
            vram_available_gb=vram_available,
            is_gpu_accelerated=False,
            confidence=1.0,
            is_measured=False,
            similar_benchmark=None,
            warnings=warnings,
        )

    # Check for existing benchmark
    benchmark = ocr_queries.get_ocr_benchmark(hardware_id, tool_id, document_type)
    is_measured = False
    similar_benchmark = None
    confidence = 0.7  # Default for estimates

    if benchmark:
        # Use real benchmark data
        is_measured = True
        pages_per_second = benchmark.pages_per_second
        ram_required = benchmark.ram_usage_gb
        vram_required = benchmark.vram_usage_gb
        is_gpu_accelerated = benchmark.is_gpu_accelerated
        confidence = 0.95
    else:
        # Estimate based on similar hardware
        similar = ocr_queries.get_similar_ocr_benchmark(
            tool_id, hw.memory_bandwidth_gbs
        )

        if similar:
            sim_bench, sim_hw_name, sim_bw = similar
            # Scale by bandwidth ratio
            bw_ratio = hw.memory_bandwidth_gbs / sim_bw
            pages_per_second = sim_bench.pages_per_second * (bw_ratio ** 0.5)
            ram_required = sim_bench.ram_usage_gb
            vram_required = sim_bench.vram_usage_gb
            is_gpu_accelerated = sim_bench.is_gpu_accelerated
            similar_benchmark = {
                "hardware_name": sim_hw_name,
                "pages_per_second": sim_bench.pages_per_second,
            }
            confidence = 0.75
        else:
            # Fallback estimate based on tool type
            if tool.tool_type == "traditional":
                # CPU-based, ~0.1-0.2 pages/sec baseline
                pages_per_second = 0.15
            elif tool.tool_type == "neural":
                if is_gpu_accelerated:
                    # GPU neural, scale by bandwidth
                    pages_per_second = hw.memory_bandwidth_gbs / 50
                else:
                    # CPU neural, slower
                    pages_per_second = 0.5
            else:
                pages_per_second = 1.0
            confidence = 0.5

    # Calculate total time
    total_time_seconds = page_count / pages_per_second if pages_per_second > 0 else 0

    # Add document type warning
    if document_type == "pdf_scanned":
        warnings.append("Scanned PDFs may be slower due to image processing")

    return OcrSimulationResult(
        can_run=True,
        pages_per_second=round(pages_per_second, 2),
        total_time_seconds=round(total_time_seconds, 1),
        ram_required_gb=ram_required,
        ram_available_gb=hw.memory_gb,
        vram_required_gb=vram_required,
        vram_available_gb=vram_available,
        is_gpu_accelerated=is_gpu_accelerated,
        confidence=confidence,
        is_measured=is_measured,
        similar_benchmark=similar_benchmark,
        warnings=warnings,
    )
