"""Simulation service for hardware/model performance estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..db import queries
from ..db.models import Hardware
from ..predictor.model import predictor


@dataclass
class CustomHardware:
    """Custom hardware specification from user input."""

    memory_gb: float
    memory_bandwidth_gbs: float
    memory_type: str  # "unified" or "vram"
    gpu_cores: Optional[int] = None
    name: Optional[str] = None


@dataclass
class CustomModel:
    """Custom model specification from user input."""

    size_gb: float
    bits_per_weight: float = 4.5
    is_moe: bool = False


@dataclass
class NearestBenchmarkInfo:
    """Nearest benchmark for confidence visualization."""

    hardware_name: str
    model_name: str
    quant_type: str
    generation_tps: float
    ttft_ms: float
    source: Optional[str]


@dataclass
class SimulationResult:
    """Result of a simulation."""

    can_run: bool
    memory_required_gb: float
    memory_available_gb: float
    generation_tps: float
    prompt_tps: Optional[float]
    ttft_ms: float
    confidence: float
    is_measured: bool
    similar_benchmark: Optional[Dict]
    nearest_benchmarks: List[NearestBenchmarkInfo]
    warnings: List[str]


def simulate(
    hardware_id: Optional[int],
    custom_hardware: Optional[CustomHardware],
    quantization_id: Optional[int],
    prompt_tokens: int,
    custom_model: Optional[CustomModel] = None,
) -> SimulationResult:
    """
    Simulate LLM performance on given hardware.

    Args:
        hardware_id: ID of predefined hardware, or None if custom
        custom_hardware: Custom hardware specs, or None if using predefined
        quantization_id: ID of model quantization to simulate, or None if custom
        prompt_tokens: Number of input prompt tokens
        custom_model: Custom model specs, or None if using predefined

    Returns:
        SimulationResult with performance estimates
    """
    warnings: list[str] = []

    # Get hardware specs
    if hardware_id:
        hw = queries.get_hardware_by_id(hardware_id)
        if not hw:
            raise ValueError(f"Hardware not found: {hardware_id}")
        memory_gb = hw.memory_gb
        bandwidth_gbs = hw.memory_bandwidth_gbs
        is_unified = hw.memory_type == "unified"
        gpu_cores = hw.gpu_cores
    elif custom_hardware:
        memory_gb = custom_hardware.memory_gb
        bandwidth_gbs = custom_hardware.memory_bandwidth_gbs
        is_unified = custom_hardware.memory_type == "unified"
        gpu_cores = custom_hardware.gpu_cores
        hw = None
    else:
        raise ValueError("Either hardware_id or custom_hardware must be provided")

    # Get model/quantization info
    if custom_model:
        model_size_gb = custom_model.size_gb
        bits_per_weight = custom_model.bits_per_weight
        is_moe = custom_model.is_moe
        quant = None
        model = None
    elif quantization_id:
        quant = queries.get_quantization_by_id(quantization_id)
        if not quant:
            raise ValueError(f"Quantization not found: {quantization_id}")
        model = queries.get_model_by_id(quant.model_id)
        if not model:
            raise ValueError(f"Model not found: {quant.model_id}")
        model_size_gb = quant.size_gb
        bits_per_weight = quant.bits_per_weight
        is_moe = model.architecture == "moe"
    else:
        raise ValueError("Either quantization_id or custom_model must be provided")

    # Calculate KV cache size based on model size and context length
    # Formula: kv_cache_gb ≈ (model_params_b / 7) * (context_length / 4096) * 0.5
    # We estimate model_params_b from size_gb and bits_per_weight
    estimated_params_b = (model_size_gb * 8) / bits_per_weight  # rough estimate
    kv_cache_gb = (estimated_params_b / 7) * (prompt_tokens / 4096) * 0.5

    # Check if model fits in memory
    memory_required = model_size_gb + kv_cache_gb
    can_run = memory_required <= memory_gb

    if not can_run:
        return SimulationResult(
            can_run=False,
            memory_required_gb=round(memory_required, 1),
            memory_available_gb=memory_gb,
            generation_tps=0,
            prompt_tps=None,
            ttft_ms=0,
            confidence=1.0,
            is_measured=False,
            similar_benchmark=None,
            nearest_benchmarks=[],
            warnings=["Model does not fit in memory"],
        )

    # Check for existing benchmark
    similar_benchmark = None
    is_measured = False
    generation_tps = 0.0
    prompt_tps = None
    ttft_ms = 0.0
    confidence = 0.0

    if hardware_id and quantization_id:
        bench = queries.get_benchmark_for_config(
            hardware_id, quantization_id, prompt_tokens
        )
        if bench:
            # Use real benchmark data
            is_measured = True
            generation_tps = bench.generation_tps
            prompt_tps = bench.prompt_tps
            ttft_ms = bench.ttft_ms if bench.ttft_ms else 0
            confidence = 0.95

            if not ttft_ms and prompt_tps:
                from ..predictor.model import estimate_ttft
                ttft_ms = estimate_ttft(prompt_tps)

    if not is_measured:
        # Use predictor
        result = predictor.predict(
            memory_bandwidth_gbs=bandwidth_gbs,
            memory_gb=memory_gb,
            model_size_gb=model_size_gb,
            bits_per_weight=bits_per_weight,
            is_unified_memory=is_unified,
            is_moe=is_moe,
            prompt_tokens=prompt_tokens,
            gpu_cores=gpu_cores,
        )

        generation_tps = result.generation_tps
        prompt_tps = result.prompt_tps
        ttft_ms = result.ttft_ms
        confidence = result.confidence

        # Try to find similar benchmark for reference (only for predefined models)
        if quantization_id:
            similar = queries.get_similar_benchmark(quantization_id, bandwidth_gbs)
            if similar:
                bench, sim_hw = similar
                similar_benchmark = {
                    "hardware_name": sim_hw.name,
                    "generation_tps": bench.generation_tps,
                }

    # Add warnings based on configuration
    if memory_required > memory_gb * 0.9:
        warnings.append("Memory usage is very high, may cause slowdowns")

    if is_moe:
        warnings.append("MoE model predictions are less accurate")

    if custom_model:
        warnings.append("Custom model - estimates only")

    # Get nearest benchmarks for confidence visualization (by hardware similarity)
    model_family = model.family if model else None
    nearest_benchmarks_raw = queries.get_nearest_benchmarks(
        target_bandwidth_gbs=bandwidth_gbs,
        target_memory_gb=memory_gb,
        target_model_size_gb=model_size_gb,
        target_model_family=model_family,
        limit=4,
    )
    nearest_benchmarks = [
        NearestBenchmarkInfo(
            hardware_name=nb_hw.name,
            model_name=nb_model_name,
            quant_type=nb_quant_type,
            generation_tps=nb_bench.generation_tps,
            ttft_ms=nb_bench.ttft_ms,
            source=nb_bench.source,
        )
        for nb_bench, nb_hw, nb_model_name, nb_quant_type in nearest_benchmarks_raw
    ]

    return SimulationResult(
        can_run=True,
        memory_required_gb=round(memory_required, 1),
        memory_available_gb=memory_gb,
        generation_tps=generation_tps,
        prompt_tps=prompt_tps,
        ttft_ms=ttft_ms,
        confidence=confidence,
        is_measured=is_measured,
        similar_benchmark=similar_benchmark,
        nearest_benchmarks=nearest_benchmarks,
        warnings=warnings,
    )
