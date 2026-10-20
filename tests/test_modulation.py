"""Unit tests for digital modulation module."""

import pytest
import numpy as np
from python_src.modulation import (
    get_constellation,
    modulate_bits,
    demodulate_symbols,
    theoretical_ber_awgn,
    theoretical_ber_rayleigh,
    MODULATION_ORDERS,
)


@pytest.mark.parametrize("mod_type,expected_m", [
    ("BPSK", 2),
    ("QPSK", 4),
    ("16-QAM", 16),
    ("64-QAM", 64),
])
def test_constellation_generation(mod_type, expected_m):
    points, bit_to_sym, sym_to_bit = get_constellation(mod_type)
    assert len(points) == expected_m
    assert len(bit_to_sym) == expected_m
    assert len(sym_to_bit) == expected_m

    # Verify unit average symbol power: E[|s|^2] == 1.0 within numerical tolerance
    avg_power = np.mean(np.abs(points) ** 2)
    assert np.isclose(avg_power, 1.0, atol=1e-5)


@pytest.mark.parametrize("mod_type", ["BPSK", "QPSK", "16-QAM", "64-QAM"])
def test_noiseless_modulation_roundtrip(mod_type):
    rng = np.random.default_rng(123)
    num_bits = 600  # Multiple of 1, 2, 4, 6
    tx_bits = rng.integers(0, 2, size=num_bits)

    symbols, pad_len = modulate_bits(tx_bits, mod_type)
    rx_bits = demodulate_symbols(symbols, mod_type, pad_len=pad_len)

    assert len(rx_bits) == num_bits
    np.testing.assert_array_equal(tx_bits, rx_bits)


def test_theoretical_ber_monotonic():
    snr_grid = np.array([0, 5, 10, 15, 20])
    for mod in ["BPSK", "QPSK", "16-QAM", "64-QAM"]:
        ber_awgn = theoretical_ber_awgn(snr_grid, mod)
        ber_ray = theoretical_ber_rayleigh(snr_grid, mod)

        # BER must strictly decrease with increasing SNR
        assert np.all(np.diff(ber_awgn) < 0)
        assert np.all(np.diff(ber_ray) < 0)
