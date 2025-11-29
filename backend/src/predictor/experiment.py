"""Experiment with different ML approaches for better prediction."""

from __future__ import annotations

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.connection import init_database, get_cursor
from src.db.seed import seed_database


def load_data():
    """Load benchmark data."""
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

    X = []
    y = []

    for row in rows:
        features = [
            row["memory_bandwidth_gbs"],
            row["memory_gb"],
            row["gpu_cores"] or 0,
            1.0 if row["memory_type"] == "unified" else 0.0,
            row["model_size_gb"],
            row["bits_per_weight"],
            1.0 if row["architecture"] == "moe" else 0.0,
            row["context_length"],
        ]
        X.append(features)
        y.append(row["generation_tps"])

    return np.array(X), np.array(y)


def add_engineered_features(X):
    """Add derived features based on domain knowledge."""
    X_new = []
    for row in X:
        bw, mem, cores, unified, model_size, bits, moe, ctx = row

        # Theoretical max TPS: bandwidth / model_size
        theoretical_tps = bw / max(model_size, 0.1)

        # Memory efficiency: how much of memory is used by model
        mem_utilization = model_size / max(mem, 1)

        # Bandwidth per GB of model
        bw_per_gb = bw / max(model_size, 0.1)

        # Log transforms for skewed features
        log_bw = np.log1p(bw)
        log_model_size = np.log1p(model_size)
        log_ctx = np.log1p(ctx)

        # Cores per model GB
        cores_per_gb = cores / max(model_size, 0.1)

        X_new.append([
            *row,  # Original features
            theoretical_tps,
            mem_utilization,
            bw_per_gb,
            log_bw,
            log_model_size,
            log_ctx,
            cores_per_gb,
        ])

    return np.array(X_new)


def test_algorithms(X, y):
    """Test different ML algorithms."""
    print("=" * 60)
    print("Testing different algorithms")
    print("=" * 60)

    algorithms = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "SVR (RBF)": SVR(kernel='rbf', C=100, gamma='scale'),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for name, model in algorithms.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        mean_score = scores.mean()
        std_score = scores.std()
        results.append((name, mean_score, std_score))
        print(f"{name:20s}: R² = {mean_score:.3f} (+/- {std_score:.3f})")

    return sorted(results, key=lambda x: x[1], reverse=True)


def test_with_engineered_features(X, y):
    """Test with additional engineered features."""
    print("\n" + "=" * 60)
    print("Testing with engineered features")
    print("=" * 60)

    X_eng = add_engineered_features(X)
    print(f"Original features: {X.shape[1]}, Engineered: {X_eng.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)

    algorithms = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    }

    for name, model in algorithms.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        print(f"{name:20s}: R² = {scores.mean():.3f} (+/- {scores.std():.3f})")


def test_polynomial_features(X, y):
    """Test with polynomial features."""
    print("\n" + "=" * 60)
    print("Testing with polynomial features (degree=2)")
    print("=" * 60)

    for name, model in [
        ("Ridge + Poly2", Ridge(alpha=1.0)),
        ("GradientBoosting + Poly2", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', model)
        ])
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        print(f"{name:25s}: R² = {scores.mean():.3f} (+/- {scores.std():.3f})")


def hyperparameter_tuning(X, y):
    """Tune hyperparameters for best models."""
    print("\n" + "=" * 60)
    print("Hyperparameter tuning (GradientBoosting)")
    print("=" * 60)

    X_eng = add_engineered_features(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Quick search with fewer combinations
    param_grid_small = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.1, 0.15],
        'min_samples_split': [3, 5],
    }

    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid_small,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_scaled, y)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best R²: {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_, scaler, X_eng


def test_log_target(X, y):
    """Test with log-transformed target."""
    print("\n" + "=" * 60)
    print("Testing with log-transformed target")
    print("=" * 60)

    y_log = np.log1p(y)

    X_eng = add_engineered_features(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    scores = cross_val_score(model, X_scaled, y_log, cv=5, scoring='r2')
    print(f"GradientBoosting (log target): R² = {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Also test with ExtraTrees
    model2 = ExtraTreesRegressor(n_estimators=200, random_state=42)
    scores2 = cross_val_score(model2, X_scaled, y_log, cv=5, scoring='r2')
    print(f"ExtraTrees (log target): R² = {scores2.mean():.3f} (+/- {scores2.std():.3f})")


def analyze_errors(X, y):
    """Analyze prediction errors."""
    print("\n" + "=" * 60)
    print("Error analysis")
    print("=" * 60)

    X_eng = add_engineered_features(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)
    errors = y - predictions
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / y) * 100

    print(f"Mean Absolute Error: {abs_errors.mean():.2f} tok/s")
    print(f"Median Absolute Error: {np.median(abs_errors):.2f} tok/s")
    print(f"Mean Percentage Error: {pct_errors.mean():.1f}%")
    print(f"Median Percentage Error: {np.median(pct_errors):.1f}%")

    # Find worst predictions
    worst_idx = np.argsort(abs_errors)[-5:]
    print("\nWorst 5 predictions:")
    for idx in worst_idx:
        print(f"  Actual: {y[idx]:.1f}, Predicted: {predictions[idx]:.1f}, Error: {errors[idx]:.1f}")


def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} tok/s\n")

    # Test different algorithms
    results = test_algorithms(X, y)

    # Test with engineered features
    test_with_engineered_features(X, y)

    # Test polynomial features
    test_polynomial_features(X, y)

    # Test log-transformed target
    test_log_target(X, y)

    # Hyperparameter tuning
    best_model, scaler, X_eng = hyperparameter_tuning(X, y)

    # Error analysis
    analyze_errors(X, y)

    print("\n" + "=" * 60)
    print("SUMMARY: Best approaches")
    print("=" * 60)
    print("Top 3 algorithms:")
    for i, (name, score, std) in enumerate(results[:3], 1):
        print(f"  {i}. {name}: R² = {score:.3f}")


if __name__ == "__main__":
    main()
