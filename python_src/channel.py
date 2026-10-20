"""Wireless Channel Models for OFDM Simulation.

Implements:
- Additive White Gaussian Noise (AWGN) with calibrated variance.
- Rayleigh fading multipath channel with configurable delay taps and power-delay profile.
- Flat Rayleigh channel.
- Carrier Frequency Offset (CFO) and fractional timing errors.
- True channel impulse response (CIR) and channel frequency response (CFR).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class WirelessChannel:
    """Configurable wireless channel supporting AWGN and Rayleigh Multipath."""

    def __init__(
        self,
        channel_type: str = "Rayleigh",
        num_taps: int = 4,
        delays: Optional[list[int]] = None,
        decay_factor: float = 0.5,
        cfo_normalized: float = 0.0,
        timing_offset: int = 0,
        random_state: Optional[int] = None,
    ):
        """Initialize channel parameters.

        Args:
            channel_type: 'AWGN' or 'Rayleigh'.
            num_taps: Number of multipath taps (for Rayleigh multipath).
            delays: Discrete sample delays for each tap [tau_0, tau_1, ...].
            decay_factor: Power decay factor for exponential delay profile (P(l) ~ exp(-decay * l)).
            cfo_normalized: Carrier Frequency Offset normalized to subcarrier spacing (Delta f / Delta f_sub).
            timing_offset: Timing offset in integer samples.
            random_state: Seed for reproducibility.
        """
        self.channel_type = channel_type
        self.num_taps = max(1, num_taps)
        self.decay_factor = decay_factor
        self.cfo_normalized = cfo_normalized
        self.timing_offset = timing_offset
        self.rng = np.random.default_rng(random_state)

        if delays is None:
            # Default linearly spaced integer delays: 0, 1, 2, ...
            self.delays = list(range(self.num_taps))
        else:
            self.delays = list(delays)
            self.num_taps = len(self.delays)

        self.last_cir: Optional[np.ndarray] = None
        self.last_cfr: Optional[np.ndarray] = None

    def generate_cir(self, max_delay: Optional[int] = None) -> np.ndarray:
        """Generate discrete-time Channel Impulse Response (CIR) h[n].

        For AWGN: h[n] = delta[n].
        For Rayleigh: taps are complex Gaussian with exponential power profile, normalized sum(|h|^2) = 1.
        """
        if self.channel_type.upper() == "AWGN":
            h = np.zeros(1, dtype=complex)
            h[0] = 1.0 + 0.0j
            self.last_cir = h
            return h

        # Rayleigh multipath
        max_d = max(self.delays) if max_delay is None else max(max_delay, max(self.delays))
        h = np.zeros(max_d + 1, dtype=complex)

        # Exponential power delay profile
        powers = np.exp(-self.decay_factor * np.arange(self.num_taps))
        powers /= np.sum(powers)  # normalize total average power to 1

        for i, delay in enumerate(self.delays):
            # Complex Gaussian tap: (X + jY) * sqrt(P/2)
            alpha_real = self.rng.normal(0, np.sqrt(powers[i] / 2.0))
            alpha_imag = self.rng.normal(0, np.sqrt(powers[i] / 2.0))
            h[delay] += alpha_real + 1j * alpha_imag

        # Ensure total energy of this realization is normalized to unity
        norm_factor = np.sqrt(np.sum(np.abs(h) ** 2))
        if norm_factor > 1e-12:
            h = h / norm_factor

        self.last_cir = h
        return h

    def compute_cfr(self, n_fft: int) -> np.ndarray:
        """Compute the true Channel Frequency Response (CFR) across N_FFT subcarriers.

        H[k] = sum_{n} h[n] * exp(-j * 2*pi * k * n / N_FFT)
        """
        if self.last_cir is None:
            self.generate_cir()

        # Compute N-point FFT of impulse response
        self.last_cfr = np.fft.fft(self.last_cir, n=n_fft)
        return self.last_cfr

    def pass_signal(
        self,
        tx_signal: np.ndarray,
        snr_db: float,
        n_fft: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pass transmitted time-domain OFDM signal through the multipath channel and AWGN.

        Args:
            tx_signal: Transmitted time-domain samples (with cyclic prefixes).
            snr_db: Target Signal-to-Noise Ratio in dB.
            n_fft: FFT size for computing CFR.

        Returns:
            rx_signal: Received noisy, faded time-domain signal.
            h_freq: True channel frequency response of length n_fft.
            noise_power: Variance of complex additive noise.
        """
        tx_signal = np.asarray(tx_signal, dtype=complex)

        # 1. Generate or update channel impulse response
        h = self.generate_cir()
        h_freq = self.compute_cfr(n_fft)

        # 2. Convolve with channel impulse response
        if self.channel_type.upper() == "AWGN":
            faded_signal = tx_signal.copy()
        else:
            # Linear convolution with CIR
            faded_signal = np.convolve(tx_signal, h, mode="full")[: len(tx_signal)]

        # 3. Apply Carrier Frequency Offset (CFO) if configured
        # Normalized CFO Delta f_norm: phase shift per sample is 2*pi * Delta f_norm / N_FFT
        if abs(self.cfo_normalized) > 1e-6:
            n_samples = len(faded_signal)
            time_idx = np.arange(n_samples)
            cfo_rot = np.exp(1j * 2.0 * np.pi * self.cfo_normalized * time_idx / n_fft)
            faded_signal = faded_signal * cfo_rot

        # 4. Apply Timing Offset if configured
        if self.timing_offset != 0:
            faded_signal = np.roll(faded_signal, self.timing_offset)

        # 5. Additive White Gaussian Noise (AWGN)
        # Signal power computation
        sig_power = np.mean(np.abs(faded_signal) ** 2)
        if sig_power < 1e-12:
            sig_power = 1.0

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_variance = sig_power / snr_linear

        # Complex Gaussian noise: CN(0, sigma_n^2) = N(0, sigma_n^2/2) + j*N(0, sigma_n^2/2)
        noise_std_per_dim = np.sqrt(noise_variance / 2.0)
        noise_real = self.rng.normal(0, noise_std_per_dim, size=len(faded_signal))
        noise_imag = self.rng.normal(0, noise_std_per_dim, size=len(faded_signal))
        noise = noise_real + 1j * noise_imag

        rx_signal = faded_signal + noise

        return rx_signal, h_freq, noise_variance
