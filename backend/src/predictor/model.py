"""Performance prediction using theoretical formulas and ML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Try to load ML model
_ml_model = None
_ml_scaler = None
_training_points = None
_max_distance = None
_use_log_transform = False

try:
    import joblib
    model_path = Path(__file__).parent.parent.parent / "data" / "predictor.pkl"
    if model_path.exists():
        data = joblib.load(model_path)
        _ml_model = data["model"]
        _ml_scaler = data["scaler"]
        _training_points = data["training_points"]
        _max_distance = data["max_distance"]
        _use_log_transform = data.get("use_log_transform", False)
        print(f"Loaded ML model from {model_path} (log_transform={_use_log_transform})")
except Exception as e:
    print(f"ML model not available, using theoretical predictions: {e}")


@dataclass
class PredictionResult:
    """Result of a performance prediction."""

    generation_tps: float
    prompt_tps: Optional[float]
    ttft_ms: float
    confidence: float
    is_measured: bool = False


def theoretical_tps(bandwidth_gbs: float, model_size_gb: float) -> float:
    """
    Calculate theoretical tokens per second based on memory bandwidth.
    Formula: tps ≈ (bandwidth * efficiency) / model_size
    """
    if model_size_gb <= 0:
        return 0.0
    efficiency = 0.85
    return (bandwidth_gbs / model_size_gb) * efficiency


def estimate_prompt_tps(
    generation_tps: float,
    is_unified_memory: bool,
    gpu_cores: Optional[int] = None,
) -> float:
    """
    Estimate prompt processing speed based on hardware characteristics.

    Ratios derived from benchmark data:
    - Apple Silicon (unified, <3000 cores): ~10x
    - DGX Spark (unified, 6000+ cores): ~50x
    - RTX 4090/5090 (dedicated, ~20000 cores): ~60-70x
    - H100/A100 (dedicated, ~15000 cores, tensor cores): ~100-150x
    """
    cores = gpu_cores or 0

    if is_unified_memory:
        # Apple Silicon: 10x base, DGX Spark (high cores): up to 50x
        if cores >= 5000:
            # DGX Spark / high-core unified systems
            ratio = 50.0
        elif cores >= 2000:
            # High-end Apple Silicon (M3/M4 Ultra)
            ratio = 15.0
        else:
            # Standard Apple Silicon
            ratio = 10.0
    else:
        # Dedicated VRAM
        if cores >= 14000:
            # Datacenter GPUs (H100: 16896, A100: 6912 but tensor cores)
            # H100 SXM shows ~150x ratio
            ratio = 120.0
        elif cores >= 10000:
            # High-end consumer (RTX 4090: 16384, RTX 5090: 21760)
            ratio = 70.0
        elif cores >= 5000:
            # Mid-range (RTX 4080, 3090)
            ratio = 50.0
        else:
            # Entry-level dedicated
            ratio = 40.0

    return generation_tps * ratio


def estimate_ttft(prompt_tps: float, typical_prompt_tokens: int = 500) -> float:
    """
    Estimate time to first token in milliseconds.

    Uses a typical prompt size (not full context window) since TTFT
    measures time to process the user's input prompt, not the entire
    context capacity.

    Args:
        prompt_tps: Prompt processing speed in tokens/second
        typical_prompt_tokens: Representative prompt size (default 500 tokens)
    """
    if prompt_tps <= 0:
        return 10000.0
    overhead_ms = 50
    processing_ms = (typical_prompt_tokens / prompt_tps) * 1000
    return overhead_ms + processing_ms


def calculate_min_requirements(model_size_gb: float, target_tps: float) -> Dict:
    """Calculate minimum hardware requirements for target performance."""
    efficiency = 0.85
    memory_overhead = 1.15

    memory_gb = model_size_gb * memory_overhead
    bandwidth_gbs = (target_tps * model_size_gb) / efficiency

    return {
        "memory_gb": round(memory_gb, 1),
        "bandwidth_gbs": round(bandwidth_gbs, 0),
    }


class Predictor:
    """Performance predictor combining theoretical formulas with ML."""

    def __init__(self):
        self.use_ml = _ml_model is not None

    def predict(
        self,
        memory_bandwidth_gbs: float,
        memory_gb: float,
        model_size_gb: float,
        bits_per_weight: float,
        is_unified_memory: bool,
        is_moe: bool,
        prompt_tokens: int,
        gpu_cores: Optional[int] = None,
    ) -> PredictionResult:
        """Predict performance for given hardware/model combination."""

        # Try ML prediction first
        # Note: prompt_tokens is passed as context_length since it represents the context size
        if self.use_ml:
            try:
                gen_tps, confidence = self._ml_predict(
                    memory_bandwidth_gbs,
                    memory_gb,
                    gpu_cores or 0,
                    is_unified_memory,
                    model_size_gb,
                    bits_per_weight,
                    is_moe,
                    prompt_tokens,  # Used as context_length
                )
            except Exception:
                gen_tps = self._theoretical_predict(
                    memory_bandwidth_gbs, model_size_gb, is_moe,
                    bits_per_weight, gpu_cores
                )
                confidence = 0.6
        else:
            gen_tps = self._theoretical_predict(
                memory_bandwidth_gbs, model_size_gb, is_moe,
                bits_per_weight, gpu_cores
            )
            confidence = self._compute_confidence(
                memory_bandwidth_gbs, model_size_gb, is_moe
            )

        prompt_tps = estimate_prompt_tps(gen_tps, is_unified_memory, gpu_cores)
        ttft_ms = estimate_ttft(prompt_tps)

        return PredictionResult(
            generation_tps=round(gen_tps, 1),
            prompt_tps=round(prompt_tps, 1),
            ttft_ms=round(ttft_ms, 0),
            confidence=confidence,
            is_measured=False,
        )

    def _ml_predict(
        self,
        memory_bandwidth_gbs: float,
        memory_gb: float,
        gpu_cores: int,
        is_unified_memory: bool,
        model_size_gb: float,
        bits_per_weight: float,
        is_moe: bool,
        context_length: int,
    ) -> tuple:
        """Use ML model for prediction."""
        bw = memory_bandwidth_gbs
        mem = memory_gb
        cores = gpu_cores
        unified = 1.0 if is_unified_memory else 0.0
        model_size = model_size_gb
        bits = bits_per_weight
        moe = 1.0 if is_moe else 0.0
        ctx = context_length

        # Engineered features (must match training)
        theoretical_tps = bw / max(model_size, 0.1)
        mem_utilization = model_size / max(mem, 1)
        bw_per_gb = bw / max(model_size, 0.1)
        log_bw = np.log1p(bw)
        log_model_size = np.log1p(model_size)
        log_ctx = np.log1p(ctx)
        cores_per_gb = cores / max(model_size, 0.1)

        features = np.array([[
            bw, mem, cores, unified, model_size, bits, moe, ctx,
            theoretical_tps, mem_utilization, bw_per_gb,
            log_bw, log_model_size, log_ctx, cores_per_gb,
        ]])

        X_scaled = _ml_scaler.transform(features)
        tps_pred = _ml_model.predict(X_scaled)[0]

        # Inverse log transform if model was trained with log target
        if _use_log_transform:
            tps = np.expm1(tps_pred)
        else:
            tps = tps_pred

        confidence = self._compute_ml_confidence(X_scaled[0])

        return max(0.5, tps), confidence

    def _compute_ml_confidence(self, x: np.ndarray) -> float:
        """Compute confidence based on distance from training data."""
        from scipy.spatial.distance import cdist
        distances = cdist([x], _training_points)
        min_dist = distances.min()
        conf = 1.0 - (min_dist / _max_distance)
        return max(0.4, min(0.92, conf))

    def _theoretical_predict(
        self,
        memory_bandwidth_gbs: float,
        model_size_gb: float,
        is_moe: bool,
        bits_per_weight: float,
        gpu_cores: Optional[int],
    ) -> float:
        """Use theoretical formula for prediction."""
        gen_tps = theoretical_tps(memory_bandwidth_gbs, model_size_gb)

        if is_moe:
            gen_tps *= 2.5

        if bits_per_weight < 4:
            gen_tps *= 0.95
        elif bits_per_weight > 8:
            gen_tps *= 0.98

        if gpu_cores and gpu_cores > 10000:
            gen_tps *= 1.05

        return gen_tps

    def _compute_confidence(
        self,
        bandwidth_gbs: float,
        model_size_gb: float,
        is_moe: bool,
    ) -> float:
        """Compute confidence score for theoretical prediction."""
        confidence = 0.70

        if bandwidth_gbs < 50 or bandwidth_gbs > 1500:
            confidence -= 0.15

        if model_size_gb > 100:
            confidence -= 0.1

        if is_moe:
            confidence -= 0.1

        return max(0.3, min(0.80, confidence))


# Global predictor instance
predictor = Predictor()
