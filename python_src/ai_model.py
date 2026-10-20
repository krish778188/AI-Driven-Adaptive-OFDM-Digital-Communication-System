"""AI Adaptation Engine for Digital Communication.

Implements:
1. Classical Threshold-Based Adaptive Modulation (Baseline).
2. Machine-Learning Adaptive Modulation (Random Forest & Gradient Boosting).
3. Explicit Objective Function: Maximize Throughput subject to BER <= BER_target.
4. Channel Uncertainty / High-Estimation-Error Fallback Guardrails.
5. Model training, evaluation, persistence (joblib), and explanation.
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from python_src.modulation import (
    MODULATION_ORDERS,
    MODULATION_INDICES,
    MODULATION_NAMES_TO_IDX,
    theoretical_ber_awgn,
    theoretical_ber_rayleigh,
)

FEATURE_COLUMNS = [
    "snr_db",
    "h_mean",
    "h_var",
    "h_min",
    "pilot_error_var",
    "target_ber_log10",
]


class ThresholdAdaptiveEngine:
    """Conventional SNR-threshold-based adaptive modulation baseline."""

    def __init__(self, target_ber: float = 1e-2):
        self.target_ber = target_ber
        self.thresholds = self._compute_thresholds(target_ber)

    def _compute_thresholds(self, target_ber: float) -> Dict[str, float]:
        """Compute empirical/analytical SNR switching thresholds for target BER."""
        # For target_ber approx 1e-2 in multipath/Rayleigh OFDM:
        # BPSK: SNR < 6 dB
        # QPSK: 6 dB <= SNR < 14 dB
        # 16-QAM: 14 dB <= SNR < 22 dB
        # 64-QAM: SNR >= 22 dB
        # Adjust dynamically based on target_ber
        offset = -3.0 * np.log10(target_ber / 1e-2)
        return {
            "BPSK_to_QPSK": max(2.0, 6.0 + offset),
            "QPSK_to_16QAM": max(8.0, 14.0 + offset),
            "16QAM_to_64QAM": max(16.0, 22.0 + offset),
        }

    def select_modulation(self, snr_db: float) -> str:
        """Select modulation based purely on SNR thresholds."""
        if snr_db < self.thresholds["BPSK_to_QPSK"]:
            return "BPSK"
        elif snr_db < self.thresholds["QPSK_to_16QAM"]:
            return "QPSK"
        elif snr_db < self.thresholds["16QAM_to_64QAM"]:
            return "16-QAM"
        else:
            return "64-QAM"


class AIAdaptiveEngine:
    """Machine-Learning Driven Adaptive Modulation Engine with Safety Guardrails."""

    def __init__(
        self,
        target_ber: float = 1e-2,
        model_type: str = "RandomForest",
        confidence_threshold: float = 0.55,
        max_pilot_error_threshold: float = 0.20,
    ):
        self.target_ber = target_ber
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.max_pilot_error_threshold = max_pilot_error_threshold

        if model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )

        self.is_trained = False
        self.feature_names = FEATURE_COLUMNS
        self.classes_ = np.array([0, 1, 2, 3])  # BPSK, QPSK, 16-QAM, 64-QAM
        self.training_metrics: Dict[str, Any] = {}

    @staticmethod
    def extract_feature_vector(
        snr_db: float,
        channel_features: Dict[str, float],
        target_ber: float = 1e-2,
    ) -> np.ndarray:
        """Extract normalized feature vector for model input."""
        h_mean = float(channel_features.get("h_mean", 1.0))
        h_var = float(channel_features.get("h_var", 0.1))
        h_min = float(channel_features.get("h_min", 0.5))
        pilot_error_var = float(channel_features.get("pilot_error_var", 0.01))
        target_ber_log10 = float(np.log10(max(target_ber, 1e-6)))

        vec = np.array([snr_db, h_mean, h_var, h_min, pilot_error_var, target_ber_log10])
        return vec.reshape(1, -1)

    def train(self, df_dataset: pd.DataFrame) -> Dict[str, Any]:
        """Train classifier using generated channel simulation dataset.

        Args:
            df_dataset: DataFrame containing feature columns and 'optimal_mod_idx'.

        Returns:
            Dict containing accuracy and classification report.
        """
        X = df_dataset[self.feature_names].values
        y = df_dataset["optimal_mod_idx"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=["BPSK", "QPSK", "16-QAM", "64-QAM"],
            output_dict=True,
            zero_division=0,
        )

        self.is_trained = True
        self.training_metrics = {
            "accuracy": acc,
            "classification_report": report,
            "num_samples": len(df_dataset),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        return self.training_metrics

    def predict_adaptation(
        self,
        snr_db: float,
        channel_features: Dict[str, float],
        target_ber: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict optimal modulation with safety fallback verification.

        Returns:
            Dict containing:
            - selected_mod: Final modulation name.
            - raw_ai_recommendation: AI model's raw recommendation.
            - confidence: Class probability confidence.
            - fallback_triggered: Boolean indicating whether fallback was activated.
            - fallback_reason: Explanation of fallback reason if triggered.
            - feature_vector: Input feature values.
        """
        target_ber = self.target_ber if target_ber is None else target_ber
        x_vec = self.extract_feature_vector(snr_db, channel_features, target_ber)

        # Fallback 1: If model is not trained yet, use Threshold baseline
        if not self.is_trained:
            th_engine = ThresholdAdaptiveEngine(target_ber)
            mod_choice = th_engine.select_modulation(snr_db)
            return {
                "selected_mod": mod_choice,
                "raw_ai_recommendation": mod_choice,
                "confidence": 1.0,
                "fallback_triggered": True,
                "fallback_reason": "AI model not trained; running baseline threshold engine.",
                "feature_vector": dict(zip(self.feature_names, x_vec[0])),
            }

        # Predict probabilities
        try:
            probs = self.model.predict_proba(x_vec)[0]
            # Align with classes [0, 1, 2, 3]
            class_probs = np.zeros(4)
            for idx, cls in enumerate(self.model.classes_):
                if 0 <= cls <= 3:
                    class_probs[cls] = probs[idx]
            raw_pred_idx = int(np.argmax(class_probs))
            confidence = float(class_probs[raw_pred_idx])
            raw_mod = MODULATION_INDICES[raw_pred_idx]
        except Exception:
            # Fallback if model is not yet fitted
            th_engine = ThresholdAdaptiveEngine(target_ber)
            raw_mod = th_engine.select_modulation(snr_db)
            raw_pred_idx = MODULATION_NAMES_TO_IDX[raw_mod]
            confidence = 0.5
            class_probs = np.zeros(4)
            class_probs[raw_pred_idx] = 1.0

        final_idx = raw_pred_idx
        fallback_triggered = False
        fallback_reason = "None (Target BER expected to be satisfied)."

        # Fallback Guardrail 1: High Channel Estimation Error
        pilot_err = float(channel_features.get("pilot_error_var", 0.0))
        if pilot_err > self.max_pilot_error_threshold and final_idx > 0:
            final_idx = max(0, final_idx - 1)
            fallback_triggered = True
            fallback_reason = (
                f"High channel estimation error ({pilot_err:.3f} > {self.max_pilot_error_threshold:.3f}); "
                f"demoting to more robust modulation {MODULATION_INDICES[final_idx]}."
            )

        # Fallback Guardrail 2: Low AI Model Confidence
        elif confidence < self.confidence_threshold and final_idx > 0:
            final_idx = max(0, final_idx - 1)
            fallback_triggered = True
            fallback_reason = (
                f"Low AI model confidence ({confidence*100:.1f}% < {self.confidence_threshold*100:.1f}%); "
                f"safely falling back to {MODULATION_INDICES[final_idx]}."
            )

        # Fallback Guardrail 3: Severe Deep Channel Null
        h_min = float(channel_features.get("h_min", 1.0))
        if h_min < 0.15 and final_idx > 1:
            final_idx = 1  # Cap at QPSK under deep channel fade
            fallback_triggered = True
            fallback_reason = (
                f"Severe multipath fade detected (min |H| = {h_min:.3f} < 0.15); "
                f"capping modulation to QPSK for link survival."
            )

        selected_mod = MODULATION_INDICES[final_idx]

        return {
            "selected_mod": selected_mod,
            "raw_ai_recommendation": raw_mod,
            "confidence": confidence,
            "probabilities": {
                MODULATION_INDICES[i]: float(class_probs[i]) for i in range(4)
            },
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "target_ber": target_ber,
            "feature_vector": dict(zip(self.feature_names, x_vec[0])),
        }

    def save(self, filepath: str, metadata_path: Optional[str] = None) -> None:
        """Persist model and metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

        if metadata_path is None:
            metadata_path = os.path.splitext(filepath)[0] + "_metadata.json"

        meta = {
            "model_type": self.model_type,
            "target_ber": self.target_ber,
            "confidence_threshold": self.confidence_threshold,
            "max_pilot_error_threshold": self.max_pilot_error_threshold,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, filepath: str, metadata_path: Optional[str] = None) -> None:
        """Load persisted model and metadata from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True

        if metadata_path is None:
            metadata_path = os.path.splitext(filepath)[0] + "_metadata.json"

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                self.model_type = meta.get("model_type", self.model_type)
                self.target_ber = meta.get("target_ber", self.target_ber)
                self.confidence_threshold = meta.get(
                    "confidence_threshold", self.confidence_threshold
                )
                self.max_pilot_error_threshold = meta.get(
                    "max_pilot_error_threshold", self.max_pilot_error_threshold
                )
                self.training_metrics = meta.get("training_metrics", {})


def generate_adaptation_dataset(
    num_samples: int = 1500,
    target_ber: float = 1e-2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic communication channel dataset for training AI adaptation model.

    Evaluates throughput of all 4 modulations across SNR and channel conditions,
    and labels optimal modulation satisfying BER <= target_ber.
    """
    from python_src.ofdm_sim import OFDMSystem
    from python_src.channel import WirelessChannel

    rng = np.random.default_rng(random_state)
    records = []

    # Fast evaluation block for dataset generation
    ofdm = OFDMSystem(n_fft=64, n_cp=16, random_state=random_state)

    for i in range(num_samples):
        # Sample SNR from -2 dB to 28 dB
        snr_db = float(rng.uniform(-2.0, 28.0))

        # Sample channel characteristics
        ch_type = rng.choice(["AWGN", "Rayleigh"], p=[0.25, 0.75])
        num_taps = int(rng.choice([1, 2, 4, 6])) if ch_type == "Rayleigh" else 1
        decay = float(rng.uniform(0.3, 0.8))

        ch = WirelessChannel(
            channel_type=ch_type,
            num_taps=num_taps,
            decay_factor=decay,
            random_state=int(rng.integers(0, 100000)),
        )

        # Transmit test probe with QPSK to extract channel features
        test_bits = ofdm.generate_random_bits(192)  # 2 OFDM symbols
        tx_sig, _, n_sym, pad = ofdm.transmit(test_bits, "QPSK")
        rx_sig, _, noise_var = ch.pass_signal(tx_sig, snr_db, n_fft=64)
        rx_res = ofdm.receive(rx_sig, "QPSK", n_sym, pad, len(test_bits), noise_var)

        feats = rx_res["channel_features"]
        h_mean = feats["h_mean"]
        h_var = feats["h_var"]
        h_min = feats["h_min"]
        pilot_error_var = feats["pilot_error_var"]

        # Objective function evaluation:
        # Determine for each modulation whether BER <= target_ber, and pick highest spectral efficiency
        # Analytical approximation calibrated with empirical fade depth:
        # Effective SNR experienced by data subcarriers
        eff_snr_lin = (10.0 ** (snr_db / 10.0)) * (h_mean ** 2) / (1.0 + 3.0 * h_var)
        eff_snr_db = 10.0 * np.log10(max(eff_snr_lin, 1e-4))

        bers = {}
        for mod in ["BPSK", "QPSK", "16-QAM", "64-QAM"]:
            if ch_type == "AWGN":
                ber_val = float(theoretical_ber_awgn(eff_snr_db, mod))
            else:
                ber_val = float(theoretical_ber_rayleigh(eff_snr_db, mod))
            bers[mod] = ber_val

        # Select highest modulation meeting BER target
        best_mod_idx = 0  # default BPSK
        max_throughput = -1.0

        for m_idx in [0, 1, 2, 3]:
            mod_name = MODULATION_INDICES[m_idx]
            b_val = bers[mod_name]
            bps = MODULATION_ORDERS[mod_name]

            # Enforce target BER constraint
            if b_val <= target_ber * 1.5:  # allow 1.5x margin
                throughput = bps * (1.0 - b_val)
                if throughput > max_throughput:
                    max_throughput = throughput
                    best_mod_idx = m_idx

        records.append({
            "snr_db": snr_db,
            "h_mean": h_mean,
            "h_var": h_var,
            "h_min": h_min,
            "pilot_error_var": pilot_error_var,
            "target_ber_log10": float(np.log10(target_ber)),
            "ber_bpsk": bers["BPSK"],
            "ber_qpsk": bers["QPSK"],
            "ber_16qam": bers["16-QAM"],
            "ber_64qam": bers["64-QAM"],
            "optimal_mod_idx": best_mod_idx,
            "optimal_mod_name": MODULATION_INDICES[best_mod_idx],
        })

    df = pd.DataFrame(records)
    return df
