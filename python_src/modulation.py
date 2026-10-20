"""Digital Modulation Module for Adaptive OFDM System.

Supports BPSK, QPSK, 16-QAM, and 64-QAM with:
- Gray-coded bit-to-symbol constellation mapping.
- Power normalization to unit average symbol energy (E_s = 1).
- Euclidean distance minimum-error symbol demapping.
- Closed-form analytical Bit Error Rate (BER) calculations for AWGN and Rayleigh fading.
"""

from __future__ import annotations
import numpy as np
from scipy import special
from typing import Tuple, Dict, Any

MODULATION_ORDERS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4,
    "64-QAM": 6,
}

MODULATION_INDICES = {
    0: "BPSK",
    1: "QPSK",
    2: "16-QAM",
    3: "64-QAM",
}

MODULATION_NAMES_TO_IDX = {v: k for k, v in MODULATION_INDICES.items()}


def get_constellation(mod_type: str) -> Tuple[np.ndarray, Dict[str, complex], Dict[complex, str]]:
    """Generate Gray-coded, unit-energy normalized constellation and mapping tables.

    Args:
        mod_type: One of 'BPSK', 'QPSK', '16-QAM', '64-QAM'.

    Returns:
        constellation: Array of complex constellation points.
        bit_to_sym: Mapping from bit string (e.g. '01') to complex symbol.
        sym_to_bit: Mapping from complex symbol to bit string.
    """
    mod_type = mod_type.upper()
    if mod_type == "BPSK":
        # 1 bit per symbol: '0' -> -1, '1' -> +1
        points = np.array([-1.0 + 0.0j, 1.0 + 0.0j])
        bit_to_sym = {"0": points[0], "1": points[1]}
        sym_to_bit = {points[0]: "0", points[1]: "1"}
        return points, bit_to_sym, sym_to_bit

    elif mod_type == "QPSK":
        # 2 bits per symbol: Gray coding [00, 01, 11, 10]
        # Gray map for I: 0 -> -1, 1 -> +1; for Q: 0 -> -1, 1 -> +1
        norm = 1.0 / np.sqrt(2.0)
        gray_map = {"0": -1.0, "1": 1.0}
        bit_to_sym = {}
        sym_to_bit = {}
        points = []
        for b0 in ["0", "1"]:
            for b1 in ["0", "1"]:
                bit_str = b0 + b1
                sym = norm * (gray_map[b0] + 1j * gray_map[b1])
                bit_to_sym[bit_str] = sym
                sym_to_bit[sym] = bit_str
                points.append(sym)
        return np.array(points), bit_to_sym, sym_to_bit

    elif mod_type == "16-QAM":
        # 4 bits per symbol: 2 bits for I, 2 bits for Q
        # Gray sequence: '00' -> -3, '01' -> -1, '11' -> +1, '10' -> +3
        norm = 1.0 / np.sqrt(10.0)
        gray_levels = {"00": -3.0, "01": -1.0, "11": 1.0, "10": 3.0}
        bit_to_sym = {}
        sym_to_bit = {}
        points = []
        for bi, vi in gray_levels.items():
            for bq, vq in gray_levels.items():
                bit_str = bi + bq
                sym = norm * (vi + 1j * vq)
                bit_to_sym[bit_str] = sym
                sym_to_bit[sym] = bit_str
                points.append(sym)
        return np.array(points), bit_to_sym, sym_to_bit

    elif mod_type == "64-QAM":
        # 6 bits per symbol: 3 bits for I, 3 bits for Q
        # Gray sequence for 3 bits:
        # '000': -7, '001': -5, '011': -3, '010': -1, '110': +1, '111': +3, '101': +5, '100': +7
        norm = 1.0 / np.sqrt(42.0)
        gray_levels = {
            "000": -7.0,
            "001": -5.0,
            "011": -3.0,
            "010": -1.0,
            "110": 1.0,
            "111": 3.0,
            "101": 5.0,
            "100": 7.0,
        }
        bit_to_sym = {}
        sym_to_bit = {}
        points = []
        for bi, vi in gray_levels.items():
            for bq, vq in gray_levels.items():
                bit_str = bi + bq
                sym = norm * (vi + 1j * vq)
                bit_to_sym[bit_str] = sym
                sym_to_bit[sym] = bit_str
                points.append(sym)
        return np.array(points), bit_to_sym, sym_to_bit

    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}. Must be BPSK, QPSK, 16-QAM, or 64-QAM.")


def modulate_bits(bits: np.ndarray, mod_type: str) -> Tuple[np.ndarray, int]:
    """Modulate a binary stream into complex baseband symbols.

    Args:
        bits: 1D array of integers in {0, 1}.
        mod_type: Modulation type ('BPSK', 'QPSK', '16-QAM', '64-QAM').

    Returns:
        symbols: 1D array of unit-energy complex symbols.
        pad_len: Number of zero-padding bits added to align with modulation order.
    """
    bps = MODULATION_ORDERS[mod_type.upper()]
    bits = np.asarray(bits, dtype=int).ravel()

    # Zero-pad if bit sequence length is not a multiple of bps
    remainder = len(bits) % bps
    pad_len = 0
    if remainder != 0:
        pad_len = bps - remainder
        bits = np.pad(bits, (0, pad_len), "constant", constant_values=0)

    _, bit_to_sym, _ = get_constellation(mod_type)

    # Group into words and map
    num_symbols = len(bits) // bps
    bit_matrix = bits.reshape((num_symbols, bps))
    symbols = np.empty(num_symbols, dtype=complex)

    for i in range(num_symbols):
        word = "".join(bit_matrix[i].astype(str))
        symbols[i] = bit_to_sym[word]

    return symbols, pad_len


def demodulate_symbols(symbols: np.ndarray, mod_type: str, pad_len: int = 0) -> np.ndarray:
    """Demodulate received complex symbols into bits using minimum Euclidean distance.

    Args:
        symbols: 1D array of complex received symbols.
        mod_type: Modulation type ('BPSK', 'QPSK', '16-QAM', '64-QAM').
        pad_len: Number of padding bits to remove from end of recovered bitstream.

    Returns:
        bits: 1D array of recovered binary integers in {0, 1}.
    """
    bps = MODULATION_ORDERS[mod_type.upper()]
    constellation, _, sym_to_bit = get_constellation(mod_type)

    symbols = np.asarray(symbols, dtype=complex).ravel()
    num_symbols = len(symbols)

    # Compute pairwise Euclidean distances to all constellation points
    # shape: (num_symbols, M)
    diffs = symbols[:, np.newaxis] - constellation[np.newaxis, :]
    dist_sq = np.real(diffs * np.conj(diffs))
    nearest_idx = np.argmin(dist_sq, axis=1)

    recovered_bits = np.empty((num_symbols, bps), dtype=int)
    for i, idx in enumerate(nearest_idx):
        point = constellation[idx]
        bit_str = sym_to_bit[point]
        recovered_bits[i] = [int(b) for b in bit_str]

    flat_bits = recovered_bits.ravel()
    if pad_len > 0:
        flat_bits = flat_bits[:-pad_len]

    return flat_bits


def q_function(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * special.erfc(np.asarray(x) / np.sqrt(2.0))


def theoretical_ber_awgn(snr_db: np.ndarray | float, mod_type: str) -> np.ndarray:
    """Calculate closed-form theoretical Bit Error Rate over AWGN channel.

    Args:
        snr_db: Signal-to-Noise Ratio (E_s/N_0 in dB).
        mod_type: Modulation type.

    Returns:
        ber: Theoretical BER values.
    """
    snr_lin = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    mod_type = mod_type.upper()
    bps = MODULATION_ORDERS[mod_type]

    # Convert E_s/N_0 to E_b/N_0: (E_b/N_0) = (E_s/N_0) / bps
    eb_n0 = snr_lin / bps

    if mod_type == "BPSK":
        # P_b = Q(sqrt(2 * E_b/N_0))
        return q_function(np.sqrt(2.0 * eb_n0))

    elif mod_type == "QPSK":
        # P_b = Q(sqrt(2 * E_b/N_0))
        return q_function(np.sqrt(2.0 * eb_n0))

    elif mod_type == "16-QAM":
        # P_b \approx (3/8) * Q(sqrt( (4/5) * E_b/N_0 ))
        return (3.0 / 8.0) * q_function(np.sqrt((4.0 / 5.0) * eb_n0))

    elif mod_type == "64-QAM":
        # P_b \approx (7/24) * Q(sqrt( (6/21) * E_b/N_0 ))
        return (7.0 / 24.0) * q_function(np.sqrt((6.0 / 21.0) * eb_n0))

    else:
        raise ValueError(f"Unknown modulation: {mod_type}")


def theoretical_ber_rayleigh(snr_db: np.ndarray | float, mod_type: str) -> np.ndarray:
    """Calculate closed-form theoretical BER over flat Rayleigh fading channel.

    Args:
        snr_db: Average SNR (E_s/N_0 in dB).
        mod_type: Modulation type.

    Returns:
        ber: Theoretical BER values.
    """
    snr_lin = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    mod_type = mod_type.upper()
    bps = MODULATION_ORDERS[mod_type]
    gamma_b = snr_lin / bps  # average Eb/N0

    if mod_type in ["BPSK", "QPSK"]:
        # P_b = 0.5 * (1 - sqrt(gamma_b / (1 + gamma_b)))
        return 0.5 * (1.0 - np.sqrt(gamma_b / (1.0 + gamma_b)))

    elif mod_type == "16-QAM":
        # Approximation for M-QAM in Rayleigh fading
        # P_b \approx (3/8) * (1 - sqrt( (0.4 * gamma_b) / (1 + 0.4 * gamma_b) ))
        c = 0.4 * gamma_b
        return (3.0 / 8.0) * (1.0 - np.sqrt(c / (1.0 + c)))

    elif mod_type == "64-QAM":
        # P_b \approx (7/24) * (1 - sqrt( (1/7 * gamma_b) / (1 + 1/7 * gamma_b) ))
        c = (1.0 / 7.0) * gamma_b
        return (7.0 / 24.0) * (1.0 - np.sqrt(c / (1.0 + c)))

    else:
        raise ValueError(f"Unknown modulation: {mod_type}")
