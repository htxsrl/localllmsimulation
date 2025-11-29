#!/usr/bin/env python3
"""Test simulation accuracy against real benchmarks."""

import sqlite3
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def get_db_connection():
    """Connect to in-memory populated database via API queries."""
    return None  # We'll use the API instead

def get_benchmarks_with_details():
    """Get 10 diverse benchmarks with hardware and model details."""
    # Query database directly for benchmarks
    import sys
    sys.path.insert(0, '/Users/franz/Desktop/SCRIVANIA/00-HTX-2024/09-Progetti/2025-11-28-LLMSimulator/backend')

    from src.db.connection import init_database, get_cursor
    from src.db.seed import seed_database

    init_database()
    seed_database()

    with get_cursor() as cursor:
        # Get diverse benchmarks (different hardware types, models, context lengths)
        cursor.execute("""
            SELECT
                b.id, b.hardware_id, b.quantization_id,
                b.generation_tps as real_gen_tps,
                b.prompt_tps as real_prompt_tps,
                b.context_length,
                h.name as hw_name, h.vendor, h.memory_type, h.gpu_cores,
                m.family, m.variant, m.size_b,
                q.quant_type
            FROM benchmarks b
            JOIN hardware h ON b.hardware_id = h.id
            JOIN quantizations q ON b.quantization_id = q.id
            JOIN models m ON q.model_id = m.id
            WHERE b.generation_tps IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 10
        """)
        return cursor.fetchall()

def get_near_benchmarks():
    """Get 10 configurations near existing benchmarks (different context or similar hw)."""
    import sys
    sys.path.insert(0, '/Users/franz/Desktop/SCRIVANIA/00-HTX-2024/09-Progetti/2025-11-28-LLMSimulator/backend')

    from src.db.connection import init_database, get_cursor
    from src.db.seed import seed_database

    init_database()
    seed_database()

    with get_cursor() as cursor:
        # Get configurations that are similar but not exact matches
        # Using hardware with similar specs but different IDs
        cursor.execute("""
            SELECT DISTINCT
                h.id as hardware_id,
                q.id as quantization_id,
                8192 as context_length,  -- Use a common context length
                h.name as hw_name, h.vendor, h.memory_type, h.gpu_cores,
                m.family, m.variant, m.size_b,
                q.quant_type,
                (SELECT COUNT(*) FROM benchmarks b
                 WHERE b.hardware_id = h.id AND b.quantization_id = q.id) as has_benchmark
            FROM hardware h
            CROSS JOIN quantizations q
            JOIN models m ON q.model_id = m.id
            WHERE q.size_gb < h.memory_gb * 0.9  -- Model must fit
            AND (SELECT COUNT(*) FROM benchmarks b
                 WHERE b.hardware_id = h.id AND b.quantization_id = q.id) = 0  -- No exact benchmark
            ORDER BY RANDOM()
            LIMIT 10
        """)
        return cursor.fetchall()

def simulate(hardware_id, quantization_id, context_length):
    """Run simulation via API."""
    response = requests.post(f"{BASE_URL}/simulate", json={
        "hardware_id": hardware_id,
        "quantization_id": quantization_id,
        "prompt_tokens": context_length or 4096
    })
    return response.json()

def main():
    print("=" * 80)
    print("PART 1: EXACT BENCHMARK COMPARISONS (Simulation vs Real)")
    print("=" * 80)
    print()

    benchmarks = get_benchmarks_with_details()

    errors_gen = []
    errors_prompt = []

    print(f"{'Hardware':<30} {'Model':<25} {'Ctx':<6} {'Real Gen':<10} {'Sim Gen':<10} {'Error %':<10} {'Real PP':<10} {'Sim PP':<10} {'PP Err %':<10}")
    print("-" * 140)

    for b in benchmarks:
        (bench_id, hw_id, quant_id, real_gen, real_prompt, ctx_len,
         hw_name, vendor, mem_type, gpu_cores,
         family, variant, size_b, quant_type) = b

        ctx = ctx_len or 4096
        sim = simulate(hw_id, quant_id, ctx)

        sim_gen = sim.get('generation_tps', 0)
        sim_prompt = sim.get('prompt_tps', 0)

        gen_error = ((sim_gen - real_gen) / real_gen * 100) if real_gen else 0
        errors_gen.append(abs(gen_error))

        prompt_error = 0
        if real_prompt and sim_prompt:
            prompt_error = ((sim_prompt - real_prompt) / real_prompt * 100)
            errors_prompt.append(abs(prompt_error))

        model_name = f"{family} {size_b}B {quant_type}"[:24]

        print(f"{hw_name[:29]:<30} {model_name:<25} {ctx:<6} {real_gen:<10.1f} {sim_gen:<10.1f} {gen_error:>+8.1f}%  {str(real_prompt or '-'):<10} {sim_prompt:<10.1f} {prompt_error:>+8.1f}%")

    print("-" * 140)
    avg_gen_error = sum(errors_gen) / len(errors_gen) if errors_gen else 0
    avg_prompt_error = sum(errors_prompt) / len(errors_prompt) if errors_prompt else 0
    print(f"Average absolute error: Generation = {avg_gen_error:.1f}%, Prompt Processing = {avg_prompt_error:.1f}%")
    print()

    print("=" * 80)
    print("PART 2: NEAR-BENCHMARK CONFIGURATIONS (Predictions)")
    print("=" * 80)
    print()

    near_configs = get_near_benchmarks()

    print(f"{'Hardware':<30} {'Model':<25} {'Ctx':<6} {'Sim Gen':<10} {'Sim PP':<10} {'TTFT ms':<10} {'Conf %':<10}")
    print("-" * 110)

    for n in near_configs:
        (hw_id, quant_id, ctx_len,
         hw_name, vendor, mem_type, gpu_cores,
         family, variant, size_b, quant_type, has_bench) = n

        sim = simulate(hw_id, quant_id, ctx_len)

        model_name = f"{family} {size_b}B {quant_type}"[:24]

        if sim.get('can_run', False):
            print(f"{hw_name[:29]:<30} {model_name:<25} {ctx_len:<6} {sim.get('generation_tps', 0):<10.1f} {sim.get('prompt_tps', 0):<10.1f} {sim.get('ttft_ms', 0):<10.0f} {sim.get('confidence', 0)*100:<10.0f}")
        else:
            print(f"{hw_name[:29]:<30} {model_name:<25} {ctx_len:<6} {'--':<10} {'--':<10} {'--':<10} {'N/A':<10} (doesn't fit)")

    print()
    print("=" * 80)
    print("PART 3: SPECIFIC KNOWN CONFIGURATIONS")
    print("=" * 80)
    print()

    # Test some specific well-known configurations
    known_tests = [
        # (hardware_id, quant_id, context, expected_gen, description)
        (97, 291, 8192, 27, "H100 SXM + Llama 70B Q4_K_M @ 8K"),
        (97, 291, 2048, 32, "H100 SXM + Llama 70B Q4_K_M @ 2K"),
        (1, 3, 4096, 43, "M4 Max 64GB + Llama 8B Q4_K_M @ 4K"),
    ]

    print(f"{'Description':<45} {'Expected':<10} {'Simulated':<10} {'Error %':<10}")
    print("-" * 80)

    for hw_id, q_id, ctx, expected, desc in known_tests:
        sim = simulate(hw_id, q_id, ctx)
        sim_gen = sim.get('generation_tps', 0)
        error = ((sim_gen - expected) / expected * 100) if expected else 0
        print(f"{desc:<45} {expected:<10.1f} {sim_gen:<10.1f} {error:>+8.1f}%")

if __name__ == "__main__":
    main()
