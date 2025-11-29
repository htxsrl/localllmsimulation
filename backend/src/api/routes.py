"""API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..db import queries, ocr_queries
from ..services import search, simulation, ocr_simulation
from . import schemas

router = APIRouter()


# Hardware endpoints


@router.get("/hardware", response_model=List[schemas.HardwareResponse])
def list_hardware(
    vendor: Optional[str] = None,
    min_memory_gb: Optional[float] = None,
    max_memory_gb: Optional[float] = None,
):
    """List all hardware, optionally filtered."""
    hardware_list = queries.get_all_hardware(
        vendor=vendor,
        min_memory_gb=min_memory_gb,
        max_memory_gb=max_memory_gb,
    )
    return [
        schemas.HardwareResponse(
            id=hw.id,
            name=hw.name,
            vendor=hw.vendor,
            chip=hw.chip,
            memory_gb=hw.memory_gb,
            memory_bandwidth_gbs=hw.memory_bandwidth_gbs,
            memory_type=hw.memory_type,
            gpu_cores=hw.gpu_cores,
        )
        for hw in hardware_list
    ]


@router.get("/hardware/{hardware_id}", response_model=schemas.HardwareDetailResponse)
def get_hardware(hardware_id: int):
    """Get hardware by ID."""
    hw = queries.get_hardware_by_id(hardware_id)
    if not hw:
        raise HTTPException(status_code=404, detail="Hardware not found")

    benchmarks_count = queries.get_benchmark_count_for_hardware(hardware_id)

    return schemas.HardwareDetailResponse(
        id=hw.id,
        name=hw.name,
        vendor=hw.vendor,
        chip=hw.chip,
        memory_gb=hw.memory_gb,
        memory_bandwidth_gbs=hw.memory_bandwidth_gbs,
        memory_type=hw.memory_type,
        gpu_cores=hw.gpu_cores,
        benchmarks_count=benchmarks_count,
    )


# Model endpoints


@router.get("/models", response_model=List[schemas.ModelResponse])
def list_models(
    family: Optional[str] = None,
    min_size_b: Optional[float] = None,
    max_size_b: Optional[float] = None,
):
    """List all models, optionally filtered."""
    models_list = queries.get_all_models(
        family=family,
        min_size_b=min_size_b,
        max_size_b=max_size_b,
    )
    return [
        schemas.ModelResponse(
            id=m.id,
            family=m.family,
            variant=m.variant,
            size_b=m.size_b,
            architecture=m.architecture,
        )
        for m in models_list
    ]


@router.get("/models/{model_id}", response_model=schemas.ModelDetailResponse)
def get_model(model_id: int):
    """Get model by ID with quantizations."""
    model = queries.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    quants = queries.get_quantizations_for_model(model_id)

    return schemas.ModelDetailResponse(
        id=model.id,
        family=model.family,
        variant=model.variant,
        size_b=model.size_b,
        architecture=model.architecture,
        quantizations=[
            schemas.QuantizationResponse(
                id=q.id,
                quant_type=q.quant_type,
                size_gb=q.size_gb,
                bits_per_weight=q.bits_per_weight,
            )
            for q in quants
        ],
    )


@router.get("/quantizations/{quant_id}", response_model=schemas.QuantizationDetailResponse)
def get_quantization(quant_id: int):
    """Get quantization by ID with model info."""
    quant = queries.get_quantization_by_id(quant_id)
    if not quant:
        raise HTTPException(status_code=404, detail="Quantization not found")

    model = queries.get_model_by_id(quant.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return schemas.QuantizationDetailResponse(
        id=quant.id,
        quant_type=quant.quant_type,
        size_gb=quant.size_gb,
        bits_per_weight=quant.bits_per_weight,
        model=schemas.ModelResponse(
            id=model.id,
            family=model.family,
            variant=model.variant,
            size_b=model.size_b,
            architecture=model.architecture,
        ),
    )


# Simulation endpoint


@router.post("/simulate", response_model=schemas.SimulationResponse)
def run_simulation(request: schemas.SimulationRequest):
    """Simulate LLM performance on hardware."""
    if request.hardware_id is None and request.custom_hardware is None:
        raise HTTPException(
            status_code=400,
            detail="Either hardware_id or custom_hardware must be provided",
        )

    if request.quantization_id is None and request.custom_model is None:
        raise HTTPException(
            status_code=400,
            detail="Either quantization_id or custom_model must be provided",
        )

    custom_hw = None
    if request.custom_hardware:
        custom_hw = simulation.CustomHardware(
            memory_gb=request.custom_hardware.memory_gb,
            memory_bandwidth_gbs=request.custom_hardware.memory_bandwidth_gbs,
            memory_type=request.custom_hardware.memory_type,
            gpu_cores=request.custom_hardware.gpu_cores,
        )

    custom_model = None
    if request.custom_model:
        custom_model = simulation.CustomModel(
            size_gb=request.custom_model.size_gb,
            bits_per_weight=request.custom_model.bits_per_weight,
            is_moe=request.custom_model.is_moe,
        )

    try:
        result = simulation.simulate(
            hardware_id=request.hardware_id,
            custom_hardware=custom_hw,
            quantization_id=request.quantization_id,
            prompt_tokens=request.prompt_tokens,
            custom_model=custom_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    similar = None
    if result.similar_benchmark:
        similar = schemas.SimilarBenchmark(
            hardware_name=result.similar_benchmark["hardware_name"],
            generation_tps=result.similar_benchmark["generation_tps"],
        )

    nearest = [
        schemas.NearestBenchmark(
            hardware_name=nb.hardware_name,
            model_name=nb.model_name,
            quant_type=nb.quant_type,
            generation_tps=nb.generation_tps,
            ttft_ms=nb.ttft_ms,
            source=nb.source,
        )
        for nb in result.nearest_benchmarks
    ]

    return schemas.SimulationResponse(
        can_run=result.can_run,
        memory_required_gb=result.memory_required_gb,
        memory_available_gb=result.memory_available_gb,
        generation_tps=result.generation_tps,
        prompt_tps=result.prompt_tps,
        ttft_ms=result.ttft_ms,
        confidence=result.confidence,
        is_measured=result.is_measured,
        similar_benchmark=similar,
        nearest_benchmarks=nearest,
        warnings=result.warnings,
    )


# Search endpoints


@router.get("/search/hardware-for-model", response_model=schemas.HardwareSearchResponse)
def search_hardware_for_model(
    quantization_id: int = Query(..., description="Quantization ID"),
    min_tps: float = Query(..., description="Minimum tokens per second"),
):
    """Find hardware that can run a model at desired speed."""
    try:
        result = search.search_hardware_for_model(quantization_id, min_tps)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return schemas.HardwareSearchResponse(
        min_requirements=schemas.MinRequirements(
            memory_gb=result.min_requirements["memory_gb"],
            bandwidth_gbs=result.min_requirements["bandwidth_gbs"],
        ),
        compatible=[
            schemas.CompatibleHardware(
                hardware=schemas.HardwareInfo(**c["hardware"]),
                estimated_tps=c["estimated_tps"],
                meets_target=c["meets_target"],
                is_measured=c["is_measured"],
                confidence=c["confidence"],
            )
            for c in result.compatible
        ],
        incompatible=[
            schemas.IncompatibleHardware(
                hardware=schemas.HardwareInfo(**i["hardware"]),
                reason=i["reason"],
                shortfall_gb=i["shortfall_gb"],
            )
            for i in result.incompatible
        ],
    )


@router.get("/search/models-for-hardware", response_model=schemas.ModelSearchResponse)
def search_models_for_hardware(
    hardware_id: Optional[int] = None,
    memory_gb: Optional[float] = None,
    bandwidth_gbs: Optional[float] = None,
    min_tps: Optional[float] = None,
):
    """Find models that can run on given hardware."""
    if hardware_id is None and (memory_gb is None or bandwidth_gbs is None):
        raise HTTPException(
            status_code=400,
            detail="Either hardware_id or both memory_gb and bandwidth_gbs required",
        )

    try:
        result = search.search_models_for_hardware(
            hardware_id=hardware_id,
            memory_gb=memory_gb,
            bandwidth_gbs=bandwidth_gbs,
            min_tps=min_tps,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return schemas.ModelSearchResponse(
        models=[
            schemas.ModelSearchItem(
                model=schemas.ModelInfo(**m["model"]),
                quantization=schemas.QuantInfo(**m["quantization"]),
                estimated_tps=m["estimated_tps"],
                can_run=m["can_run"],
                confidence=m["confidence"],
            )
            for m in result.models
        ]
    )


# OCR endpoints


@router.get("/ocr-tools", response_model=List[schemas.OcrToolResponse])
def list_ocr_tools():
    """List all OCR tools."""
    tools = ocr_queries.get_all_ocr_tools()
    return [
        schemas.OcrToolResponse(
            id=t.id,
            name=t.name,
            vendor=t.vendor,
            tool_type=t.tool_type,
            requires_gpu=t.requires_gpu,
            min_vram_gb=t.min_vram_gb,
            base_ram_gb=t.base_ram_gb,
            description=t.description,
        )
        for t in tools
    ]


@router.post("/simulate-ocr", response_model=schemas.OcrSimulationResponse)
def run_ocr_simulation(request: schemas.OcrSimulationRequest):
    """Simulate OCR/document parsing performance."""
    if request.document_type not in ("pdf_text", "pdf_scanned", "mixed"):
        raise HTTPException(
            status_code=400,
            detail="document_type must be 'pdf_text', 'pdf_scanned', or 'mixed'",
        )

    try:
        result = ocr_simulation.simulate_ocr(
            hardware_id=request.hardware_id,
            tool_id=request.tool_id,
            page_count=request.page_count,
            document_type=request.document_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    similar = None
    if result.similar_benchmark:
        similar = schemas.OcrSimilarBenchmark(
            hardware_name=result.similar_benchmark["hardware_name"],
            pages_per_second=result.similar_benchmark["pages_per_second"],
        )

    return schemas.OcrSimulationResponse(
        can_run=result.can_run,
        pages_per_second=result.pages_per_second,
        total_time_seconds=result.total_time_seconds,
        ram_required_gb=result.ram_required_gb,
        ram_available_gb=result.ram_available_gb,
        vram_required_gb=result.vram_required_gb,
        vram_available_gb=result.vram_available_gb,
        is_gpu_accelerated=result.is_gpu_accelerated,
        confidence=result.confidence,
        is_measured=result.is_measured,
        similar_benchmark=similar,
        warnings=result.warnings,
    )


# Benchmarks list endpoint


@router.get("/benchmarks", response_model=List[schemas.BenchmarkListItem])
def list_benchmarks():
    """List all benchmarks for the sources table."""
    return queries.get_all_benchmarks()


# Model info endpoint


@router.get("/model-info", response_model=schemas.MLModelInfoResponse)
def get_ml_model_info():
    """Get information about the ML prediction model."""
    import joblib
    from pathlib import Path

    model_path = Path(__file__).parent.parent.parent / "data" / "predictor.pkl"

    if not model_path.exists():
        return schemas.MLModelInfoResponse(
            algorithm="Theoretical Formula",
            r2_score=None,
            mean_absolute_error=None,
            mean_percentage_error=None,
            training_samples=0,
            feature_importance=[],
            use_log_transform=False,
        )

    data = joblib.load(model_path)
    model = data["model"]
    metrics = data.get("metrics", {})

    # Get feature importance
    features = data.get("features", [])
    importances = model.feature_importances_ if hasattr(model, "feature_importances_") else []

    feature_importance = []
    if len(features) == len(importances):
        for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:10]:
            feature_importance.append(schemas.FeatureImportance(
                name=name,
                importance=round(float(imp) * 100, 1),
            ))

    return schemas.MLModelInfoResponse(
        algorithm=type(model).__name__,
        r2_score=metrics.get("r2_score"),
        mean_absolute_error=metrics.get("mean_absolute_error"),
        mean_percentage_error=metrics.get("mean_percentage_error"),
        training_samples=len(data.get("training_points", [])),
        feature_importance=feature_importance,
        use_log_transform=data.get("use_log_transform", False),
    )
