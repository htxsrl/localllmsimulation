"""Search service for finding hardware/models that meet requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..db import queries
from ..predictor.model import calculate_min_requirements, predictor


@dataclass
class HardwareSearchResult:
    """Result of searching for compatible hardware."""

    min_requirements: Dict
    compatible: List[Dict]
    incompatible: List[Dict]


@dataclass
class ModelSearchResult:
    """Result of searching for compatible models."""

    models: List[Dict]


def search_hardware_for_model(
    quantization_id: int,
    min_tps: float,
) -> HardwareSearchResult:
    """
    Find hardware that can run a model at desired speed.

    Args:
        quantization_id: Model quantization to search for
        min_tps: Minimum required tokens per second

    Returns:
        HardwareSearchResult with compatible and incompatible hardware
    """
    # Get quantization and model info
    quant = queries.get_quantization_by_id(quantization_id)
    if not quant:
        raise ValueError(f"Quantization not found: {quantization_id}")

    model = queries.get_model_by_id(quant.model_id)
    if not model:
        raise ValueError(f"Model not found: {quant.model_id}")

    # Calculate minimum requirements
    min_reqs = calculate_min_requirements(quant.size_gb, min_tps)

    # Get all hardware
    all_hardware = queries.get_all_hardware()

    compatible = []
    incompatible = []

    for hw in all_hardware:
        # Check memory
        memory_required = quant.size_gb * 1.1
        if hw.memory_gb < memory_required:
            incompatible.append({
                "hardware": {
                    "id": hw.id,
                    "name": hw.name,
                    "vendor": hw.vendor,
                    "chip": hw.chip,
                    "memory_gb": hw.memory_gb,
                    "memory_bandwidth_gbs": hw.memory_bandwidth_gbs,
                },
                "reason": "memory",
                "shortfall_gb": round(memory_required - hw.memory_gb, 1),
            })
            continue

        # Predict performance
        result = predictor.predict(
            memory_bandwidth_gbs=hw.memory_bandwidth_gbs,
            memory_gb=hw.memory_gb,
            model_size_gb=quant.size_gb,
            bits_per_weight=quant.bits_per_weight,
            is_unified_memory=hw.memory_type == "unified",
            is_moe=model.architecture == "moe",
            prompt_tokens=500,  # Standard benchmark prompt
            gpu_cores=hw.gpu_cores,
        )

        # Check for real benchmark
        bench = queries.get_benchmark_for_config(hw.id, quantization_id)
        if bench:
            estimated_tps = bench.generation_tps
            is_measured = True
            confidence = 0.95
        else:
            estimated_tps = result.generation_tps
            is_measured = False
            confidence = result.confidence

        meets_target = estimated_tps >= min_tps

        if estimated_tps >= min_tps * 0.8:  # Include if within 20% of target
            compatible.append({
                "hardware": {
                    "id": hw.id,
                    "name": hw.name,
                    "vendor": hw.vendor,
                    "chip": hw.chip,
                    "memory_gb": hw.memory_gb,
                    "memory_bandwidth_gbs": hw.memory_bandwidth_gbs,
                },
                "estimated_tps": round(estimated_tps, 1),
                "meets_target": meets_target,
                "is_measured": is_measured,
                "confidence": confidence,
            })
        else:
            incompatible.append({
                "hardware": {
                    "id": hw.id,
                    "name": hw.name,
                    "vendor": hw.vendor,
                    "chip": hw.chip,
                    "memory_gb": hw.memory_gb,
                    "memory_bandwidth_gbs": hw.memory_bandwidth_gbs,
                },
                "reason": "speed",
                "shortfall_gb": None,
            })

    # Sort compatible by tps descending
    compatible.sort(key=lambda x: x["estimated_tps"], reverse=True)

    return HardwareSearchResult(
        min_requirements=min_reqs,
        compatible=compatible,
        incompatible=incompatible,
    )


def search_models_for_hardware(
    hardware_id: Optional[int] = None,
    memory_gb: Optional[float] = None,
    bandwidth_gbs: Optional[float] = None,
    min_tps: Optional[float] = None,
) -> ModelSearchResult:
    """
    Find models that can run on given hardware.

    Args:
        hardware_id: Predefined hardware ID, or None if custom
        memory_gb: Custom memory size (if hardware_id is None)
        bandwidth_gbs: Custom bandwidth (if hardware_id is None)
        min_tps: Optional minimum performance requirement

    Returns:
        ModelSearchResult with compatible models
    """
    # Get hardware specs
    if hardware_id:
        hw = queries.get_hardware_by_id(hardware_id)
        if not hw:
            raise ValueError(f"Hardware not found: {hardware_id}")
        memory_gb = hw.memory_gb
        bandwidth_gbs = hw.memory_bandwidth_gbs
        is_unified = hw.memory_type == "unified"
        gpu_cores = hw.gpu_cores
    else:
        if memory_gb is None or bandwidth_gbs is None:
            raise ValueError("memory_gb and bandwidth_gbs required for custom hardware")
        is_unified = True  # Assume unified for custom
        gpu_cores = None
        hw = None

    # Get all models with quantizations
    all_models = queries.get_all_models()

    results = []

    for model in all_models:
        quants = queries.get_quantizations_for_model(model.id)

        for quant in quants:
            # Check if it fits
            memory_required = quant.size_gb * 1.1
            can_run = memory_required <= memory_gb

            if not can_run:
                continue

            # Predict performance
            result = predictor.predict(
                memory_bandwidth_gbs=bandwidth_gbs,
                memory_gb=memory_gb,
                model_size_gb=quant.size_gb,
                bits_per_weight=quant.bits_per_weight,
                is_unified_memory=is_unified,
                is_moe=model.architecture == "moe",
                prompt_tokens=500,
                gpu_cores=gpu_cores,
            )

            estimated_tps = result.generation_tps
            confidence = result.confidence

            # Check benchmark if hardware_id provided
            if hardware_id:
                bench = queries.get_benchmark_for_config(hardware_id, quant.id)
                if bench:
                    estimated_tps = bench.generation_tps
                    confidence = 0.95

            # Filter by min_tps if specified
            if min_tps and estimated_tps < min_tps:
                continue

            results.append({
                "model": {
                    "id": model.id,
                    "family": model.family,
                    "variant": model.variant,
                    "size_b": model.size_b,
                },
                "quantization": {
                    "id": quant.id,
                    "quant_type": quant.quant_type,
                    "size_gb": quant.size_gb,
                },
                "estimated_tps": round(estimated_tps, 1),
                "can_run": True,
                "confidence": confidence,
            })

    # Sort by estimated_tps descending
    results.sort(key=lambda x: x["estimated_tps"], reverse=True)

    return ModelSearchResult(models=results)
