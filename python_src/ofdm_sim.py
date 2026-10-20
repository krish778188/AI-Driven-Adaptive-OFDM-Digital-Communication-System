"""OFDM Simulation Engine (Transmitter, Channel, Receiver, Equalizer).

Features:
- IEEE 802.11a/g style subcarrier allocation:
  - N_fft = 64 subcarriers.
  - 52 active subcarriers (48 data subcarriers, 4 pilot subcarriers).
  - DC null subcarrier and 11 guard subcarriers.
  - Pilot subcarriers at indices [-21, -7, 7, 21].
  - Cyclic prefix of length N_cp = 16.
- Transmitter:
  - Serial-to-parallel conversion.
  - Multi-scheme modulation (BPSK, QPSK, 16-QAM, 64-QAM).
  - Subcarrier grid assembly with known pilot sequences.
  - Normalized IFFT modulation.
  - Cyclic Prefix insertion.
- Receiver:
  - Cyclic Prefix removal.
  - Normalized FFT demodulation.
  - Pilot extraction and Least Squares (LS) channel estimation.
  - 1D linear channel frequency response interpolation.
  - Zero-Forcing (ZF) and MMSE equalizers.
  - Constellation demapping and binary data recovery.
  - BER, EVM, and channel metric extraction.
"""

from __future__ import annotations
import numpy as np
from scipy import interpolate
from typing import Tuple, Dict, Any, Optional

from python_src.modulation import (
    modulate_bits,
    demodulate_symbols,
    MODULATION_ORDERS,
    get_constellation,
)
from python_src.channel import WirelessChannel


class OFDMSystem:
    """End-to-end OFDM communication system simulator."""

    def __init__(
        self,
        n_fft: int = 64,
        n_cp: int = 16,
        pilot_subcarriers: Optional[list[int]] = None,
        data_subcarriers: Optional[list[int]] = None,
        equalizer_type: str = "MMSE",  # 'ZF' or 'MMSE'
        random_state: Optional[int] = None,
    ):
        """Initialize OFDM system parameters.

        Args:
            n_fft: Total number of subcarriers (default 64).
            n_cp: Cyclic prefix length (default 16).
            pilot_subcarriers: Subcarrier indices for pilot symbols.
            data_subcarriers: Subcarrier indices for data symbols.
            equalizer_type: 'MMSE' or 'ZF'.
            random_state: Random seed for data generation.
        """
        self.n_fft = n_fft
        self.n_cp = n_cp
        self.equalizer_type = equalizer_type.upper()
        self.rng = np.random.default_rng(random_state)

        # Standard 802.11-like active subcarrier layout (indices 0..N_fft-1)
        # In shifted coordinates (center DC = 0):
        # Active indices: [-26..-1, 1..26] (52 subcarriers)
        # Pilot indices: [-21, -7, 7, 21] (4 pilots)
        # Data indices: remaining 48 active subcarriers
        if pilot_subcarriers is None or data_subcarriers is None:
            active_shifted = np.array([k for k in range(-26, 27) if k != 0])
            pilots_shifted = np.array([-21, -7, 7, 21])
            data_shifted = np.array([k for k in active_shifted if k not in pilots_shifted])

            # Convert to standard FFT indexing [0..N_fft-1]
            self.pilot_idx = np.sort((pilots_shifted + n_fft) % n_fft)
            self.data_idx = np.sort((data_shifted + n_fft) % n_fft)
        else:
            self.pilot_idx = np.array(pilot_subcarriers, dtype=int)
            self.data_idx = np.array(data_subcarriers, dtype=int)

        self.n_pilots = len(self.pilot_idx)
        self.n_data = len(self.data_idx)

        # Predefine known deterministic BPSK pilot values (+1, -1, +1, +1)
        self.pilot_values = np.array([1.0, -1.0, 1.0, 1.0][: self.n_pilots], dtype=complex)
        if len(self.pilot_values) < self.n_pilots:
            self.pilot_values = np.ones(self.n_pilots, dtype=complex)

    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        """Generate independent and identically distributed binary bits {0, 1}."""
        return self.rng.integers(0, 2, size=num_bits, dtype=int)

    def transmit(
        self,
        bits: np.ndarray,
        mod_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Modulate binary data and synthesize time-domain OFDM signal.

        Args:
            bits: Input binary bitstream.
            mod_type: Modulation scheme ('BPSK', 'QPSK', '16-QAM', '64-QAM').

        Returns:
            tx_time_signal: Serialized time-domain baseband samples with CP.
            tx_data_symbols: Complex data symbols before OFDM subcarrier mapping.
            num_ofdm_symbols: Number of OFDM symbols transmitted.
            pad_len: Number of padding bits added.
        """
        bps = MODULATION_ORDERS[mod_type.upper()]

        # Modulate input bits to complex symbols
        tx_data_symbols, pad_len = modulate_bits(bits, mod_type)

        # Compute required number of OFDM symbols
        num_data_syms = len(tx_data_symbols)
        num_ofdm_symbols = int(np.ceil(num_data_syms / self.n_data))

        # Pad symbol stream if needed to fill all OFDM symbols completely
        pad_syms_needed = (num_ofdm_symbols * self.n_data) - num_data_syms
        if pad_syms_needed > 0:
            tx_data_symbols = np.pad(
                tx_data_symbols, (0, pad_syms_needed), "constant", constant_values=0.0 + 0.0j
            )

        # Reshape data symbols into (num_ofdm_symbols, n_data)
        data_matrix = tx_data_symbols.reshape((num_ofdm_symbols, self.n_data))

        # Build OFDM subcarrier grid: (num_ofdm_symbols, N_fft)
        freq_grid = np.zeros((num_ofdm_symbols, self.n_fft), dtype=complex)
        freq_grid[:, self.data_idx] = data_matrix
        freq_grid[:, self.pilot_idx] = self.pilot_values

        # IFFT modulation (unitary normalization with 1/sqrt(N_fft))
        # Note: np.fft.ifft uses 1/N, so we multiply by sqrt(N_fft) to normalize energy
        time_blocks = np.fft.ifft(freq_grid, axis=1) * np.sqrt(self.n_fft)

        # Cyclic prefix insertion: copy last n_cp samples to beginning of each block
        cp = time_blocks[:, -self.n_cp :]
        time_blocks_with_cp = np.hstack([cp, time_blocks])

        # Serialize into 1D continuous time-domain signal
        tx_time_signal = time_blocks_with_cp.ravel()

        return tx_time_signal, tx_data_symbols, num_ofdm_symbols, pad_len

    def receive(
        self,
        rx_time_signal: np.ndarray,
        mod_type: str,
        num_ofdm_symbols: int,
        pad_len: int,
        orig_bit_len: int,
        noise_variance: float = 1e-4,
    ) -> Dict[str, Any]:
        """Process received time-domain OFDM signal through demodulation and equalization.

        Args:
            rx_time_signal: Received time-domain samples.
            mod_type: Modulation scheme.
            num_ofdm_symbols: Number of transmitted OFDM symbols.
            pad_len: Padding bits added at transmitter.
            orig_bit_len: Original number of transmitted bits before padding.
            noise_variance: Estimated channel noise power for MMSE equalizer.

        Returns:
            Dict containing recovered bits, equalized symbols, estimated CFR, EVM, and metrics.
        """
        symbol_len = self.n_fft + self.n_cp
        total_samples = num_ofdm_symbols * symbol_len

        # Truncate or zero-pad received signal to expected length
        if len(rx_time_signal) < total_samples:
            rx_time_signal = np.pad(
                rx_time_signal, (0, total_samples - len(rx_time_signal)), "constant"
            )
        else:
            rx_time_signal = rx_time_signal[:total_samples]

        # Reshape into OFDM symbol blocks: (num_ofdm_symbols, symbol_len)
        rx_blocks_with_cp = rx_time_signal.reshape((num_ofdm_symbols, symbol_len))

        # 1. Cyclic Prefix Removal
        rx_blocks = rx_blocks_with_cp[:, self.n_cp :]

        # 2. FFT Demodulation (unitary normalization)
        rx_freq_grid = np.fft.fft(rx_blocks, axis=1) / np.sqrt(self.n_fft)

        # 3. Pilot Extraction & Least-Squares Channel Estimation
        rx_pilots = rx_freq_grid[:, self.pilot_idx]
        # H_hat_pilots = Y_pilots / X_pilots
        h_hat_pilots = rx_pilots / self.pilot_values  # shape: (num_ofdm_symbols, n_pilots)

        # Average pilot channel estimates across symbols for static/block fading
        avg_h_hat_pilots = np.mean(h_hat_pilots, axis=0)

        # 4. Channel Interpolation across all subcarriers
        # Linear interpolation wrapped across circular subcarrier indices
        pilot_idx_sorted = np.sort(self.pilot_idx)
        interp_func = interpolate.interp1d(
            pilot_idx_sorted,
            avg_h_hat_pilots,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        all_subcarrier_indices = np.arange(self.n_fft)
        h_hat_all = interp_func(all_subcarrier_indices)

        # Extract estimated CFR at data subcarriers
        h_hat_data = h_hat_all[self.data_idx]

        # 5. Channel Equalization
        rx_data = rx_freq_grid[:, self.data_idx]  # shape: (num_ofdm_symbols, n_data)

        if self.equalizer_type == "ZF":
            # Zero-Forcing: X_hat = Y / H_hat (with small epsilon to avoid div by zero)
            eps = 1e-8
            h_safe = np.where(np.abs(h_hat_data) < eps, eps, h_hat_data)
            equalized_data = rx_data / h_safe
        else:
            # MMSE: X_hat = (H_hat* / (|H_hat|^2 + sigma_n^2)) * Y
            h_conj = np.conj(h_hat_data)
            h_power = np.abs(h_hat_data) ** 2
            mmse_weight = h_conj / (h_power + max(noise_variance, 1e-6))
            equalized_data = rx_data * mmse_weight

        # Flatten equalized data symbols
        equalized_symbols = equalized_data.ravel()

        # 6. Demapping & Bit Recovery
        recovered_bits = demodulate_symbols(equalized_symbols, mod_type, pad_len=pad_len)

        # Trim to original transmitted bit length
        if len(recovered_bits) > orig_bit_len:
            recovered_bits = recovered_bits[:orig_bit_len]

        # 7. Channel Feature Extraction
        # Pilot residual error (proxy for channel estimation accuracy)
        # Re-estimate pilots using interpolated CFR and compare to received pilots
        h_interp_pilots = h_hat_all[self.pilot_idx]
        pilot_error_var = float(np.mean(np.abs(avg_h_hat_pilots - h_interp_pilots) ** 2))

        # Channel frequency selectivity features
        h_mag = np.abs(h_hat_data)
        h_var = float(np.var(h_mag))
        h_min = float(np.min(h_mag))
        h_mean = float(np.mean(h_mag))

        return {
            "recovered_bits": recovered_bits,
            "equalized_symbols": equalized_symbols,
            "rx_freq_grid": rx_freq_grid,
            "h_hat_all": h_hat_all,
            "h_hat_data": h_hat_data,
            "pilot_error_var": pilot_error_var,
            "channel_features": {
                "h_mean": h_mean,
                "h_var": h_var,
                "h_min": h_min,
                "pilot_error_var": pilot_error_var,
            },
        }

    def run_simulation(
        self,
        mod_type: str,
        snr_db: float,
        channel: WirelessChannel,
        num_bits: int = 10000,
    ) -> Dict[str, Any]:
        """Execute a complete end-to-end simulation run.

        Args:
            mod_type: Modulation type ('BPSK', 'QPSK', '16-QAM', '64-QAM').
            snr_db: Channel SNR in dB.
            channel: Initialized WirelessChannel instance.
            num_bits: Number of random binary bits to transmit.

        Returns:
            Dict containing BER, EVM, Throughput, and diagnostics.
        """
        # 1. Source bits
        tx_bits = self.generate_random_bits(num_bits)

        # 2. Transmitter
        tx_time_signal, tx_data_syms, num_ofdm_syms, pad_len = self.transmit(tx_bits, mod_type)

        # 3. Wireless Channel
        rx_time_signal, true_cfr, noise_variance = channel.pass_signal(
            tx_time_signal, snr_db, n_fft=self.n_fft
        )

        # 4. Receiver
        rx_result = self.receive(
            rx_time_signal=rx_time_signal,
            mod_type=mod_type,
            num_ofdm_symbols=num_ofdm_syms,
            pad_len=pad_len,
            orig_bit_len=len(tx_bits),
            noise_variance=noise_variance,
        )

        rx_bits = rx_result["recovered_bits"]

        # 5. Performance Metric Calculation
        # Bit Error Rate (BER)
        min_len = min(len(tx_bits), len(rx_bits))
        bit_errors = int(np.sum(tx_bits[:min_len] != rx_bits[:min_len]))
        ber = float(bit_errors / min_len) if min_len > 0 else 0.0

        # Error Vector Magnitude (EVM)
        eq_syms = rx_result["equalized_symbols"][: len(tx_data_syms)]
        ref_syms = tx_data_syms[: len(eq_syms)]
        error_vector = eq_syms - ref_syms
        evm_rms = float(np.sqrt(np.mean(np.abs(error_vector) ** 2) / np.mean(np.abs(ref_syms) ** 2)))

        # Spectral efficiency & Throughput
        bps = MODULATION_ORDERS[mod_type.upper()]
        # Overhead ratio = data_subcarriers / total_symbol_duration
        # Each OFDM symbol of length (N_fft + N_cp) carries n_data symbols
        ofdm_efficiency = (self.n_data / (self.n_fft + self.n_cp))
        spectral_efficiency = float(bps * ofdm_efficiency * (1.0 - ber))

        # Net effective throughput assuming 20 MHz channel bandwidth (like 802.11a)
        bandwidth_hz = 20e6
        subcarrier_spacing = bandwidth_hz / self.n_fft  # 312.5 kHz
        symbol_duration = (self.n_fft + self.n_cp) / bandwidth_hz  # 4 microseconds
        # Raw data rate = (n_data * bps) / symbol_duration
        raw_bitrate_mbps = (self.n_data * bps) / (symbol_duration * 1e6)
        net_throughput_mbps = float(raw_bitrate_mbps * (1.0 - ber))

        features = rx_result["channel_features"]
        features["snr_db"] = snr_db
        features["evm_rms"] = evm_rms

        return {
            "mod_type": mod_type,
            "snr_db": snr_db,
            "channel_type": channel.channel_type,
            "num_bits": min_len,
            "bit_errors": bit_errors,
            "ber": ber,
            "evm_rms": evm_rms,
            "spectral_efficiency": spectral_efficiency,
            "net_throughput_mbps": net_throughput_mbps,
            "raw_bitrate_mbps": raw_bitrate_mbps,
            "channel_features": features,
            "tx_bits_sample": tx_bits[:32].tolist(),
            "rx_bits_sample": rx_bits[:32].tolist(),
            "tx_symbols_sample": tx_data_syms[:400],
            "rx_symbols_sample": eq_syms[:400],
            "h_true_cfr": true_cfr,
            "h_est_cfr": rx_result["h_hat_all"],
            "tx_time_signal_sample": tx_time_signal[:240],
            "rx_time_signal_sample": rx_time_signal[:240],
        }
