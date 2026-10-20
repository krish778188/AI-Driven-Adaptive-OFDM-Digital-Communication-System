"""Unit tests for OFDM simulation engine and channel models."""

import pytest
import numpy as np
from python_src.ofdm_sim import OFDMSystem
from python_src.channel import WirelessChannel


def test_ofdm_block_dimensions():
    ofdm = OFDMSystem(n_fft=64, n_cp=16)
    assert ofdm.n_data == 48
    assert ofdm.n_pilots == 4

    bits = ofdm.generate_random_bits(192)  # 192 bits / 2 bps (QPSK) = 96 symbols = 2 OFDM symbols
    tx_signal, tx_syms, num_syms, pad = ofdm.transmit(bits, "QPSK")

    assert num_syms == 2
    # Each OFDM symbol length is N_fft + N_cp = 80 samples
    assert len(tx_signal) == 2 * 80


def test_ofdm_high_snr_awgn_recovery():
    """At 30 dB SNR in AWGN, observed BER for QPSK should be zero."""
    ofdm = OFDMSystem(n_fft=64, n_cp=16, random_state=42)
    channel = WirelessChannel(channel_type="AWGN", random_state=42)

    res = ofdm.run_simulation(
        mod_type="QPSK",
        snr_db=30.0,
        channel=channel,
        num_bits=4800,
    )

    assert res["bit_errors"] == 0
    assert res["ber"] == 0.0
    assert res["evm_rms"] < 0.08  # Low error vector magnitude


def test_rayleigh_multipath_equalizer_modes():
    """Test both ZF and MMSE equalizers run without errors under Rayleigh multipath."""
    channel = WirelessChannel(
        channel_type="Rayleigh",
        num_taps=4,
        decay_factor=0.5,
        random_state=101,
    )

    for eq_mode in ["ZF", "MMSE"]:
        ofdm = OFDMSystem(n_fft=64, n_cp=16, equalizer_type=eq_mode, random_state=101)
        res = ofdm.run_simulation(
            mod_type="16-QAM",
            snr_db=18.0,
            channel=channel,
            num_bits=3840,
        )
        assert res["num_bits"] == 3840
        assert 0.0 <= res["ber"] <= 1.0
        assert "h_mean" in res["channel_features"]
