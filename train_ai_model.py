"""Training and Pre-computation script for AI Adaptation Engine and Benchmark Curves.

Executes:
1. Generates 2,000 realistic simulation channel samples across AWGN and Rayleigh fading.
2. Trains and cross-validates the AI Adaptation Classifier (Random Forest).
3. Persists model to models/adaptive_rf_model.joblib with metadata.
4. Pre-computes Monte Carlo benchmark sweeps across SNR (-2 to 24 dB) for all schemes
   (Fixed BPSK, QPSK, 16-QAM, 64-QAM, Adaptive-Threshold, Adaptive-AI)
   and saves to results/benchmark_results.csv for instantaneous dashboard responsiveness.
"""

import os
import json
import time
import numpy as np
import pandas as pd

from python_src.ai_model import (
    AIAdaptiveEngine,
    generate_adaptation_dataset,
)
from python_src.simulation_interface import SimulationInterface

def main():
    print("==========================================================")
    print("  Generating AI Training Dataset & Pre-training Model    ")
    print("==========================================================")
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Dataset Generation
    t0 = time.time()
    print("Generating 2,000 diverse channel simulation records...")
    df_dataset = generate_adaptation_dataset(num_samples=2000, target_ber=1e-2, random_state=42)
    dataset_path = os.path.join("data", "simulation_dataset.csv")
    df_dataset.to_csv(dataset_path, index=False)
    print(f"Dataset generated ({len(df_dataset)} rows) in {time.time() - t0:.2f}s -> {dataset_path}")
    
    # 2. Model Training
    print("\nTraining Random Forest Adaptive Classifier...")
    ai_engine = AIAdaptiveEngine(target_ber=1e-2, model_type="RandomForest")
    metrics = ai_engine.train(df_dataset)
    
    model_path = os.path.join("models", "adaptive_rf_model.joblib")
    meta_path = os.path.join("models", "model_metadata.json")
    ai_engine.save(model_path, meta_path)
    
    print(f"Model Training Complete! Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")
    
    # 3. Benchmark Monte Carlo Sweep
    print("\n==========================================================")
    print("  Running Multi-Scheme Monte Carlo Benchmark Sweep        ")
    print("==========================================================")
    sim_iface = SimulationInterface(workspace_dir=os.getcwd())
    
    snr_grid = np.arange(0, 26, 2)
    schemes = ["BPSK", "QPSK", "16-QAM", "64-QAM", "Adaptive-Threshold", "Adaptive-AI"]
    
    print(f"Sweeping SNR range: {snr_grid[0]} to {snr_grid[-1]} dB in steps of 2 dB")
    print(f"Schemes: {schemes}")
    
    df_bench = sim_iface.run_snr_sweep(
        snr_range=snr_grid,
        mod_schemes=schemes,
        channel_type="Rayleigh",
        num_taps=4,
        num_bits=24000,
        target_ber=1e-2,
        ai_engine=ai_engine,
    )
    
    bench_path = os.path.join("results", "benchmark_results.csv")
    df_bench.to_csv(bench_path, index=False)
    print(f"\nBenchmark results successfully computed and saved to: {bench_path}")
    print("Done!")

if __name__ == "__main__":
    main()
