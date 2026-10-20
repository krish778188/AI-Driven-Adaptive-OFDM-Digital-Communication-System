"""Unit tests for AI Adaptation Engine and Threshold baseline."""

import pytest
import numpy as np
import pandas as pd
from python_src.ai_model import (
    ThresholdAdaptiveEngine,
    AIAdaptiveEngine,
    generate_adaptation_dataset,
)


def test_threshold_engine_adaptation_monotonicity():
    engine = ThresholdAdaptiveEngine(target_ber=1e-2)
    # Check that low SNR gives robust BPSK, and high SNR gives high-order 64-QAM
    assert engine.select_modulation(2.0) == "BPSK"
    assert engine.select_modulation(10.0) == "QPSK"
    assert engine.select_modulation(18.0) == "16-QAM"
    assert engine.select_modulation(26.0) == "64-QAM"


def test_ai_engine_dataset_generation_and_training():
    df = generate_adaptation_dataset(num_samples=150, target_ber=1e-2, random_state=42)
    assert len(df) == 150
    assert "optimal_mod_idx" in df.columns
    assert set(df["optimal_mod_idx"].unique()).issubset({0, 1, 2, 3})

    ai_engine = AIAdaptiveEngine(target_ber=1e-2, model_type="RandomForest")
    metrics = ai_engine.train(df)

    assert ai_engine.is_trained
    assert metrics["accuracy"] > 0.70  # Should learn cleanly


def test_ai_fallback_guardrail_trigger():
    # If channel estimation error is very high, engine should trigger safety fallback
    df = generate_adaptation_dataset(num_samples=100, target_ber=1e-2, random_state=42)
    ai_engine = AIAdaptiveEngine(target_ber=1e-2, max_pilot_error_threshold=0.15)
    ai_engine.train(df)

    # Unreliable channel with high pilot estimation error
    bad_channel_feats = {
        "h_mean": 0.8,
        "h_var": 0.3,
        "h_min": 0.1,
        "pilot_error_var": 0.35,  # Exceeds threshold of 0.15
    }

    pred = ai_engine.predict_adaptation(snr_db=20.0, channel_features=bad_channel_feats)
    assert pred["fallback_triggered"] is True
    assert "High channel estimation error" in pred["fallback_reason"]
