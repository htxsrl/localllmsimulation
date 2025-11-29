"""Train ML model for performance prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.connection import init_database, get_cursor
from src.db.seed import seed_database


FEATURES = [
    "memory_bandwidth_gbs",
    "memory_gb",
    "gpu_cores",
    "is_unified_memory",
    "model_size_gb",
    "bits_per_weight",
    "is_moe",
    "context_length",
    # Engineered features
    "theoretical_tps",
    "mem_utilization",
    "bw_per_gb",
    "log_bw",
    "log_model_size",
    "log_ctx",
    "cores_per_gb",
]


def load_training_data():
    """Load benchmark data and join with hardware/model info."""
    init_database()
    seed_database()

    with get_cursor() as cursor:
        cursor.execute("""
            SELECT
                h.memory_bandwidth_gbs,
                h.memory_gb,
                h.gpu_cores,
                h.memory_type,
                q.size_gb as model_size_gb,
                q.bits_per_weight,
                m.architecture,
                b.context_length,
                b.generation_tps
            FROM benchmarks b
            JOIN hardware h ON b.hardware_id = h.id
            JOIN quantizations q ON b.quantization_id = q.id
            JOIN models m ON q.model_id = m.id
            WHERE b.context_length IS NOT NULL
        """)
        rows = cursor.fetchall()

    if not rows:
        print("No benchmark data found!")
        return None, None

    X = []
    y = []

    for row in rows:
        bw = row["memory_bandwidth_gbs"]
        mem = row["memory_gb"]
        cores = row["gpu_cores"] or 0
        unified = 1.0 if row["memory_type"] == "unified" else 0.0
        model_size = row["model_size_gb"]
        bits = row["bits_per_weight"]
        moe = 1.0 if row["architecture"] == "moe" else 0.0
        ctx = row["context_length"]

        # Engineered features based on domain knowledge
        theoretical_tps = bw / max(model_size, 0.1)
        mem_utilization = model_size / max(mem, 1)
        bw_per_gb = bw / max(model_size, 0.1)
        log_bw = np.log1p(bw)
        log_model_size = np.log1p(model_size)
        log_ctx = np.log1p(ctx)
        cores_per_gb = cores / max(model_size, 0.1)

        features = [
            bw, mem, cores, unified, model_size, bits, moe, ctx,
            theoretical_tps, mem_utilization, bw_per_gb,
            log_bw, log_model_size, log_ctx, cores_per_gb,
        ]
        X.append(features)
        y.append(row["generation_tps"])

    return np.array(X), np.array(y)


def train_model(X, y):
    """Train an ExtraTrees model with log-transformed target.

    Returns model, scaler, and metrics dictionary.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Log-transform target for better predictions
    y_log = np.log1p(y)

    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation on log-transformed target
    scores = cross_val_score(model, X_scaled, y_log, cv=min(5, len(X)), scoring="r2")
    print(f"Cross-validation R² scores: {scores}")
    print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    # Train on all data
    model.fit(X_scaled, y_log)

    # Calculate predictions and metrics
    predictions_log = model.predict(X_scaled)
    predictions = np.expm1(predictions_log)

    abs_errors = np.abs(y - predictions)
    pct_errors = np.abs((y - predictions) / y) * 100

    metrics = {
        "r2_score": float(scores.mean()),
        "mean_absolute_error": float(abs_errors.mean()),
        "mean_percentage_error": float(pct_errors.mean()),
    }

    # Feature importance
    print("\nFeature importance:")
    for name, importance in sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:10]:  # Top 10
        print(f"  {name}: {importance:.3f}")

    return model, scaler, metrics


def compute_training_distances(X_scaled):
    """Compute distances for confidence estimation."""
    from scipy.spatial.distance import cdist
    distances = cdist(X_scaled, X_scaled)
    np.fill_diagonal(distances, np.inf)
    max_distance = np.percentile(distances[distances != np.inf], 95)
    return X_scaled, max_distance


def save_model(model, scaler, training_points, max_distance, metrics, output_path):
    """Save trained model and metadata including metrics."""
    data = {
        "model": model,
        "scaler": scaler,
        "training_points": training_points,
        "max_distance": max_distance,
        "features": FEATURES,
        "use_log_transform": True,  # Flag for inference
        "metrics": metrics,
    }
    joblib.dump(data, output_path)
    print(f"\nModel saved to {output_path}")


def main():
    print("Loading training data...")
    X, y = load_training_data()

    if X is None:
        return

    print(f"Loaded {len(X)} benchmark samples")
    print(f"TPS range: {y.min():.1f} - {y.max():.1f}")

    print("\nTraining model...")
    model, scaler, metrics = train_model(X, y)

    X_scaled = scaler.transform(X)
    training_points, max_distance = compute_training_distances(X_scaled)

    # Test predictions (inverse log transform)
    print("\nSample predictions vs actual:")
    predictions_log = model.predict(X_scaled)
    predictions = np.expm1(predictions_log)  # Inverse of log1p
    for i in range(min(5, len(X))):
        print(f"  Actual: {y[i]:.1f}, Predicted: {predictions[i]:.1f}")

    # Print metrics
    print(f"\nModel Metrics:")
    print(f"  R² Score: {metrics['r2_score']:.3f}")
    print(f"  Mean Absolute Error: {metrics['mean_absolute_error']:.2f} tok/s")
    print(f"  Mean Percentage Error: {metrics['mean_percentage_error']:.1f}%")

    # Save model
    output_path = Path(__file__).parent.parent.parent / "data" / "predictor.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, scaler, training_points, max_distance, metrics, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
