"""Interactive Plotly Visualizations for OFDM Communication and AI Adaptation.

Includes:
1. Constellation Diagrams with EVM annotation.
2. BER vs. SNR Curves with theoretical analytical benchmarks.
3. Throughput and Spectral Efficiency vs. SNR Curves.
4. True vs. Estimated Channel Frequency Response (CFR) across subcarriers.
5. OFDM Time-Domain Waveform with Cyclic Prefix demarcation.
6. AI Adaptation Probability & Decision Explanation Charts.
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional

from python_src.modulation import (
    get_constellation,
    theoretical_ber_awgn,
    theoretical_ber_rayleigh,
)

COLORS = {
    "BPSK": "#1f77b4",       # Blue
    "QPSK": "#2ca02c",       # Green
    "16-QAM": "#ff7f0e",     # Orange
    "64-QAM": "#d62728",     # Red
    "Adaptive-AI": "#9467bd",# Purple
    "Adaptive-Threshold": "#8c564b", # Brown
}


def plot_constellation(
    tx_symbols: np.ndarray,
    rx_symbols: np.ndarray,
    mod_type: str,
    evm_rms: float,
    max_points: int = 400,
) -> go.Figure:
    """Generate I/Q constellation diagram comparing transmitted and equalized symbols."""
    const_points, _, _ = get_constellation(mod_type)

    rx_sample = rx_symbols[:max_points]
    tx_sample = tx_symbols[:max_points]

    fig = go.Figure()

    # Received noisy equalized symbols
    fig.add_trace(
        go.Scatter(
            x=np.real(rx_sample),
            y=np.imag(rx_sample),
            mode="markers",
            name=f"Received Equalized ({mod_type})",
            marker=dict(
                size=5,
                color="#00bcd4",
                opacity=0.6,
                line=dict(width=0.5, color="white"),
            ),
        )
    )

    # Ideal constellation reference points
    fig.add_trace(
        go.Scatter(
            x=np.real(const_points),
            y=np.imag(const_points),
            mode="markers",
            name="Ideal Constellation Grid",
            marker=dict(
                size=12,
                color="#e91e63",
                symbol="x",
                line=dict(width=2, color="#e91e63"),
            ),
        )
    )

    fig.update_layout(
        title=f"I/Q Constellation Diagram - {mod_type} (EVM: {evm_rms*100:.2f}%)",
        xaxis_title="In-Phase (I)",
        yaxis_title="Quadrature (Q)",
        xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="#666"),
        yaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="#666"),
        template="plotly_dark",
        width=550,
        height=500,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    return fig


def plot_ber_curves(
    df_results: pd.DataFrame,
    target_ber: float = 1e-2,
    channel_type: str = "Rayleigh",
    show_theoretical: bool = True,
) -> go.Figure:
    """Generate publication-quality BER vs. SNR curves on a semi-log scale."""
    fig = go.Figure()

    # Plot empirical simulated curves
    schemes = df_results["scheme"].unique()
    for scheme in schemes:
        sub_df = df_results[df_results["scheme"] == scheme].sort_values("snr_db")
        color = COLORS.get(scheme, "#ffffff")
        line_dash = "solid" if "Adaptive" in scheme else "dot"
        line_width = 3 if "Adaptive" in scheme else 2

        # Replace 0 BER with 1e-6 for log plotting
        ber_plot = np.where(sub_df["ber"] == 0, 1e-6, sub_df["ber"])

        fig.add_trace(
            go.Scatter(
                x=sub_df["snr_db"],
                y=ber_plot,
                mode="lines+markers",
                name=f"Simulated: {scheme}",
                line=dict(color=color, width=line_width, dash=line_dash),
                marker=dict(size=6),
            )
        )

    # Theoretical AWGN / Rayleigh lines
    if show_theoretical:
        snr_dense = np.linspace(df_results["snr_db"].min(), df_results["snr_db"].max(), 60)
        for mod in ["BPSK", "16-QAM"]:
            if channel_type.upper() == "AWGN":
                ber_th = theoretical_ber_awgn(snr_dense, mod)
                name = f"Theory AWGN ({mod})"
            else:
                ber_th = theoretical_ber_rayleigh(snr_dense, mod)
                name = f"Theory Rayleigh ({mod})"

            fig.add_trace(
                go.Scatter(
                    x=snr_dense,
                    y=ber_th,
                    mode="lines",
                    name=name,
                    line=dict(color=COLORS.get(mod, "#aaa"), width=1.5, dash="dash"),
                    opacity=0.6,
                )
            )

    # Target BER horizontal threshold line
    fig.add_trace(
        go.Scatter(
            x=[df_results["snr_db"].min(), df_results["snr_db"].max()],
            y=[target_ber, target_ber],
            mode="lines",
            name=f"Target BER ({target_ber:.1e})",
            line=dict(color="#f44336", width=2, dash="dashdot"),
        )
    )

    fig.update_layout(
        title=f"Bit Error Rate (BER) vs. SNR ({channel_type} Channel)",
        xaxis_title="SNR (dB)",
        yaxis_title="Bit Error Rate (BER)",
        yaxis_type="log",
        yaxis_range=[-5, 0],
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
        height=550,
    )
    return fig


def plot_throughput_curves(df_results: pd.DataFrame) -> go.Figure:
    """Generate Effective Throughput (Mbps) and Spectral Efficiency vs. SNR."""
    fig = go.Figure()

    schemes = df_results["scheme"].unique()
    for scheme in schemes:
        sub_df = df_results[df_results["scheme"] == scheme].sort_values("snr_db")
        color = COLORS.get(scheme, "#ffffff")
        line_width = 3.5 if "Adaptive" in scheme else 2.0

        fig.add_trace(
            go.Scatter(
                x=sub_df["snr_db"],
                y=sub_df["net_throughput_mbps"],
                mode="lines+markers",
                name=f"{scheme}",
                line=dict(color=color, width=line_width),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Net Effective Throughput (Mbps) vs. SNR (20 MHz Bandwidth)",
        xaxis_title="SNR (dB)",
        yaxis_title="Net Throughput (Mbps)",
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        height=500,
    )
    return fig


def plot_channel_frequency_response(
    true_cfr: np.ndarray,
    est_cfr: np.ndarray,
    pilot_idx: list[int] | np.ndarray,
) -> go.Figure:
    """Plot True vs. Estimated Channel Frequency Response across subcarriers."""
    n_fft = len(true_cfr)
    k_axis = np.arange(n_fft)

    true_mag = 20.0 * np.log10(np.maximum(np.abs(true_cfr), 1e-4))
    est_mag = 20.0 * np.log10(np.maximum(np.abs(est_cfr), 1e-4))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=k_axis,
            y=true_mag,
            mode="lines",
            name="True Channel |H(f)|",
            line=dict(color="#00e676", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=k_axis,
            y=est_mag,
            mode="lines+markers",
            name="Estimated Channel |H_hat(f)| (LS + Interp)",
            line=dict(color="#ff9100", width=2, dash="dot"),
            marker=dict(size=4),
        )
    )

    # Highlight Pilot Subcarriers
    fig.add_trace(
        go.Scatter(
            x=pilot_idx,
            y=true_mag[pilot_idx],
            mode="markers",
            name="Pilot Subcarriers",
            marker=dict(size=10, color="#d500f9", symbol="diamond"),
        )
    )

    fig.update_layout(
        title="Channel Frequency Response (CFR) & Pilot Estimation",
        xaxis_title="Subcarrier Index (k)",
        yaxis_title="Magnitude (dB)",
        template="plotly_dark",
        legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02),
        height=450,
    )
    return fig


def plot_ofdm_waveform(
    tx_time_signal: np.ndarray,
    rx_time_signal: np.ndarray,
    n_samples: int = 160,
    n_cp: int = 16,
    n_fft: int = 64,
) -> go.Figure:
    """Plot baseband time-domain OFDM waveform with CP demarcation."""
    tx_sub = tx_time_signal[:n_samples]
    rx_sub = rx_time_signal[:n_samples]
    t = np.arange(len(tx_sub))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.real(tx_sub),
            mode="lines",
            name="Transmitted Real(x[n])",
            line=dict(color="#29b6f6", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.real(rx_sub),
            mode="lines",
            name="Received Real(y[n])",
            line=dict(color="#ab47bc", width=1.5),
            opacity=0.7,
        )
    )

    # Mark CP boundaries
    block_len = n_fft + n_cp
    for b in range(n_samples // block_len + 1):
        cp_end = b * block_len + n_cp
        if cp_end < n_samples:
            fig.add_vline(
                x=cp_end,
                line_width=1.5,
                line_dash="dash",
                line_color="#ffca28",
                annotation_text=f"CP End #{b+1}",
                annotation_position="top left",
            )

    fig.update_layout(
        title=f"OFDM Time-Domain Waveform Samples (CP = {n_cp}, FFT = {n_fft})",
        xaxis_title="Sample Index (n)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=400,
    )
    return fig


def plot_ai_probability_bar(probabilities: Dict[str, float], selected_mod: str) -> go.Figure:
    """Bar chart displaying AI model class probability distribution."""
    schemes = list(probabilities.keys())
    probs = [probabilities[s] for s in schemes]
    bar_colors = ["#00e676" if s == selected_mod else "#424242" for s in schemes]

    fig = go.Figure(
        data=[
            go.Bar(
                x=schemes,
                y=probs,
                marker_color=bar_colors,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="AI Adaptation Model Probabilities",
        xaxis_title="Modulation Scheme",
        yaxis_title="Predicted Probability",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
        height=320,
    )
    return fig
