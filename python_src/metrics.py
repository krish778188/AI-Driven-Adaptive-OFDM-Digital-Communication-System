"""Communication Metrics and Performance Evaluator.

Implements standard definitions for:
- Bit Error Rate (BER)
- Symbol Error Rate (SER)
- Frame / Block Error Rate (FER)
- Error Vector Magnitude (EVM) in % and dB
- Effective Throughput (Goodput in Mbps)
- Spectral Efficiency (bps/Hz)
- Shannon Channel Capacity
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> Dict[str, Any]:
    """Calculate Bit Error Rate (BER) between transmitted and received bitstreams."""
    min_len = min(len(tx_bits), len(rx_bits))
    if min_len == 0:
        return {"num_bits": 0, "bit_errors": 0, "ber": 0.0}

    bit_errors = int(np.sum(tx_bits[:min_len] != rx_bits[:min_len]))
    ber = float(bit_errors / min_len)

    return {
        "num_bits": min_len,
        "bit_errors": bit_errors,
        "ber": ber,
    }


def compute_evm(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> Dict[str, float]:
    """Calculate Error Vector Magnitude (EVM) in percentage and dB.

    EVM_rms = sqrt( mean(|rx - tx|^2) / mean(|tx|^2) )
    EVM_dB  = 20 * log10(EVM_rms)
    """
    min_len = min(len(tx_symbols), len(rx_symbols))
    if min_len == 0:
        return {"evm_rms": 0.0, "evm_pct": 0.0, "evm_db": -100.0}

    tx_s = tx_symbols[:min_len]
    rx_s = rx_symbols[:min_len]

    err = rx_s - tx_s
    ref_power = np.mean(np.abs(tx_s) ** 2)
    err_power = np.mean(np.abs(err) ** 2)

    if ref_power < 1e-12:
        return {"evm_rms": 0.0, "evm_pct": 0.0, "evm_db": -100.0}

    evm_rms = float(np.sqrt(err_power / ref_power))
    evm_pct = float(evm_rms * 100.0)
    evm_db = float(20.0 * np.log10(max(evm_rms, 1e-6)))

    return {
        "evm_rms": evm_rms,
        "evm_pct": evm_pct,
        "evm_db": evm_db,
    }


def compute_throughput(
    bps: int,
    n_data: int = 48,
    n_fft: int = 64,
    n_cp: int = 16,
    bandwidth_hz: float = 20e6,
    ber: float = 0.0,
) -> Dict[str, float]:
    """Compute raw bitrate, net goodput, and spectral efficiency.

    Args:
        bps: Bits per modulation symbol (1 for BPSK, 2 for QPSK, 4 for 16-QAM, 6 for 64-QAM).
        n_data: Number of data subcarriers per OFDM symbol.
        n_fft: Total FFT subcarrier size.
        n_cp: Cyclic prefix length in samples.
        bandwidth_hz: Total RF channel bandwidth in Hz (default 20 MHz).
        ber: Observed bit error rate.

    Returns:
        Dict with raw bitrate (Mbps), net throughput (Mbps), and spectral efficiency (bps/Hz).
    """
    symbol_duration = (n_fft + n_cp) / bandwidth_hz
    raw_rate_bps = (n_data * bps) / symbol_duration
    raw_rate_mbps = raw_rate_bps / 1e6

    # Net goodput accounts for packet/bit losses: R_net = R_raw * (1 - BER)
    net_throughput_mbps = raw_rate_mbps * max(0.0, (1.0 - ber))

    # Spectral efficiency = net_rate_bps / bandwidth_hz
    spectral_efficiency = (net_throughput_mbps * 1e6) / bandwidth_hz

    return {
        "raw_bitrate_mbps": float(raw_rate_mbps),
        "net_throughput_mbps": float(net_throughput_mbps),
        "spectral_efficiency": float(spectral_efficiency),
    }


def compute_shannon_capacity(snr_db: float, bandwidth_hz: float = 20e6) -> float:
    """Calculate theoretical continuous-input Shannon Channel Capacity:

    C = B * log2(1 + SNR_linear) [in Mbps].
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    capacity_bps = bandwidth_hz * np.log2(1.0 + snr_lin)
    return float(capacity_bps / 1e6)
