"""AI-Driven Adaptive OFDM Digital Communication System - Interactive Dashboard.

A Streamlit web application providing:
- Section A: System Overview & Block Architecture
- Section B: Simulation Controls (Modulation, Channel, SNR, Target BER)
- Section C: Live / Run-Based Simulation Engine
- Section D: Signal Visualizations (Constellation, CFR, Waveform, Bit Stream)
- Section E: Performance Analysis (BER vs SNR, Throughput, Spectral Efficiency)
- Section F: AI Decision Explanation & Safety Guardrails
- Section G: Experiment Reports & Data Export (CSV/JSON)
"""

from __future__ import annotations
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from python_src.ofdm_sim import OFDMSystem
from python_src.channel import WirelessChannel
from python_src.ai_model import AIAdaptiveEngine, ThresholdAdaptiveEngine
from python_src.simulation_interface import SimulationInterface
from python_src.visualization import (
    plot_constellation,
    plot_ber_curves,
    plot_throughput_curves,
    plot_channel_frequency_response,
    plot_ofdm_waveform,
    plot_ai_probability_bar,
)
from python_src.modulation import MODULATION_ORDERS

# Page configuration
st.set_page_config(
    page_title="AI-Driven Adaptive OFDM Laboratory",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Engineering Laboratory Styling
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1a1f2c;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 4px solid #00bcd4;
    }
    .status-badge-ok {
        background-color: #1b5e20;
        color: #e8f5e9;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-badge-warn {
        background-color: #e65100;
        color: #fff3e0;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .card-box {
        background-color: #161b26;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #2b3245;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system_engines():
    """Load and cache the simulation interface and pre-trained AI model."""
    workspace = os.getcwd()
    sim_iface = SimulationInterface(workspace_dir=workspace)

    model_path = os.path.join(workspace, "models", "adaptive_rf_model.joblib")
    meta_path = os.path.join(workspace, "models", "model_metadata.json")

    ai_engine = AIAdaptiveEngine()
    if os.path.exists(model_path):
        ai_engine.load(model_path, meta_path)

    return sim_iface, ai_engine


sim_iface, ai_engine = load_system_engines()

# ---------------------------------------------------------
# SIDEBAR CONTROLS (Section B)
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/satellite-sending-signal.png", width=64)
st.sidebar.title("🎛️ Simulation Controls")

st.sidebar.subheader("1. Modulation Mode")
mod_mode = st.sidebar.selectbox(
    "Transmission Scheme",
    ["Adaptive-AI", "Adaptive-Threshold", "BPSK", "QPSK", "16-QAM", "64-QAM"],
    index=0,
    help="Select fixed modulation or autonomous adaptive modulation.",
)

st.sidebar.subheader("2. Wireless Channel Parameters")
ch_type = st.sidebar.radio("Channel Type", ["Rayleigh", "AWGN"], index=0)

num_taps = 4
decay_factor = 0.5
cfo_val = 0.0

if ch_type == "Rayleigh":
    col_t1, col_t2 = st.sidebar.columns(2)
    with col_t1:
        num_taps = st.slider("Multipath Taps", 1, 8, 4)
    with col_t2:
        decay_factor = st.slider("Power Decay", 0.1, 1.2, 0.5, step=0.1)

enable_cfo = st.sidebar.checkbox("Enable Carrier Frequency Offset (CFO)", value=False)
if enable_cfo:
    cfo_val = st.sidebar.slider("Norm. CFO (Δf / Δf_sub)", -0.15, 0.15, 0.04, step=0.01)

st.sidebar.subheader("3. Link Budget & Quality")
snr_db = st.sidebar.slider("Channel SNR (dB)", -2.0, 26.0, 14.0, step=1.0)
target_ber_exp = st.sidebar.select_slider(
    "Target Quality (BER QoS)",
    options=[-4, -3, -2, -1],
    value=-2,
    format_func=lambda x: f"10^{x} ({10**x:.1e})",
)
target_ber = float(10 ** target_ber_exp)

st.sidebar.subheader("4. Execution Scope")
num_bits = st.sidebar.select_slider(
    "Transmitted Bits per Run",
    options=[5000, 10000, 20000, 40000, 80000],
    value=20000,
)
rand_seed = st.sidebar.number_input("Random Seed (Reproducibility)", value=42, step=1)


# ---------------------------------------------------------
# MAIN LABORATORY TABS
# ---------------------------------------------------------
st.title("📡 AI-Driven Adaptive OFDM Digital Communication System")
st.caption(
    "High-Performance Physical Layer Simulator, Real-Time Channel Estimation, and Machine Learning Modulation Selection"
)

tab_overview, tab_live, tab_signals, tab_analysis, tab_ai, tab_reports = st.tabs([
    "🏛️ System Overview",
    "⚡ Live Simulation",
    "📊 Signal Visualization",
    "📈 Performance Curves",
    "🤖 AI Adaptation Engine",
    "📄 Experiment Reports",
])

# ---------------------------------------------------------
# TAB 1: SYSTEM OVERVIEW (Section A)
# ---------------------------------------------------------
with tab_overview:
    st.subheader("System Architecture & Processing Pipeline")
    st.markdown(
        """
        This project implements an end-to-end **Adaptive Orthogonal Frequency Division Multiplexing (OFDM)** 
        communication chain integrated with an autonomous **Machine Learning Adaptation Engine**. The system dynamically 
        evaluates frequency-selective multipath channel conditions and selects the highest order modulation 
        scheme that satisfies a strict Quality of Service (QoS) Target Bit Error Rate constraint.
        """
    )

    col_arch1, col_arch2 = st.columns([3, 2])
    with col_arch1:
        st.markdown(
            """
            ```
            ┌────────────────────────────────────────────────────────────────────────┐
            │                     TRANSMITTER SUBSYSTEM                              │
            │  [Binary Source] ──> [S/P] ──> [Modulator] ──> [Pilot & Subcarrier Map] │
            │                              (BPSK..64QAM)    │ (52 Active, 4 Pilots)  │
            │                                               ▼                        │
            │  [Tx Waveform] <── [Add CP] <─────────── [64-point IFFT]               │
            └───────────────────────────────────┬────────────────────────────────────┘
                                                │
                                                ▼ Wireless Channel
            ┌────────────────────────────────────────────────────────────────────────┐
            │      [Multipath Rayleigh Taps] + [Normalized CFO] + [Calibrated AWGN]  │
            └───────────────────────────────────┬────────────────────────────────────┘
                                                │
                                                ▼
            ┌────────────────────────────────────────────────────────────────────────┐
            │                      RECEIVER SUBSYSTEM                                │
            │  [Remove CP] ──> [64-point FFT] ──> [Pilot LS Channel Estimation]      │
            │                                               │                        │
            │  [Recovered Bits] <── [Demap] <── [MMSE / ZF Equalizer]                │
            └────────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ Channel Quality Features (SNR, |H|, residual error)
            ┌────────────────────┴───────────────────────────────────────────────────┐
            │                  AI ADAPTATION ENGINE (Random Forest)                  │
            │  * Maximizes Throughput subject to BER <= Target BER                   │
            │  * Autonomous fallback guardrails on high channel estimation error     │
            └────────────────────────────────────────────────────────────────────────┘
            ```
            """
        )
    with col_arch2:
        st.markdown(
            """
            <div class="card-box">
            <h4>📐 System Parameter Profile</h4>
            <ul>
              <li><b>Bandwidth:</b> 20.0 MHz (802.11a/g standard)</li>
              <li><b>FFT Size (N_fft):</b> 64 subcarriers</li>
              <li><b>Subcarrier Spacing:</b> 312.5 kHz</li>
              <li><b>Active Subcarriers:</b> 52 (48 Data, 4 Pilots)</li>
              <li><b>Pilot Subcarrier Indices:</b> -21, -7, +7, +21</li>
              <li><b>Cyclic Prefix (N_cp):</b> 16 samples (0.8 µs)</li>
              <li><b>OFDM Symbol Duration:</b> 4.0 µs</li>
              <li><b>Equalization:</b> MMSE & Zero-Forcing</li>
              <li><b>Modulations:</b> BPSK, QPSK, 16-QAM, 64-QAM</li>
              <li><b>AI Classifier:</b> Random Forest (100 Estimators)</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# RUN SIMULATION / SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "last_sim_result" not in st.session_state:
    # Run initial default simulation
    init_cfg = {
        "mod_type": mod_mode,
        "snr_db": snr_db,
        "channel_type": ch_type,
        "num_taps": num_taps,
        "decay_factor": decay_factor,
        "cfo_normalized": cfo_val,
        "num_bits": num_bits,
        "target_ber": target_ber,
        "random_state": rand_seed,
    }
    st.session_state["last_sim_result"] = sim_iface.run_simulation(init_cfg, ai_engine=ai_engine)
    st.session_state["result_source"] = "Initial Live Run"

# ---------------------------------------------------------
# TAB 2: LIVE SIMULATION (Section C)
# ---------------------------------------------------------
with tab_live:
    st.subheader("⚡ Live Communication Simulation")

    col_btn1, col_btn2, col_info = st.columns([1, 1, 3])
    with col_btn1:
        run_clicked = st.button("▶️ Execute Simulation", type="primary", use_container_width=True)
    with col_btn2:
        reset_clicked = st.button("🔄 Reset Parameters", use_container_width=True)

    with col_info:
        st.markdown(
            f"**Execution Engine:** `{st.session_state['last_sim_result']['engine_used']}` | "
            f"**Data Status:** `{st.session_state.get('result_source', 'Live')}`"
        )

    if run_clicked:
        cfg = {
            "mod_type": mod_mode,
            "snr_db": snr_db,
            "channel_type": ch_type,
            "num_taps": num_taps,
            "decay_factor": decay_factor,
            "cfo_normalized": cfo_val,
            "num_bits": num_bits,
            "target_ber": target_ber,
            "random_state": rand_seed,
        }
        with st.spinner("Processing transmission, channel convolution, and equalization..."):
            sim_res = sim_iface.run_simulation(cfg, ai_engine=ai_engine)
            st.session_state["last_sim_result"] = sim_res
            st.session_state["result_source"] = "Interactive Live Run"
        st.success(f"Simulation completed in {sim_res['execution_time_sec']*1000:.1f} ms!")

    current_res = st.session_state["last_sim_result"]

    # Top KPI Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Active Modulation", current_res["mod_type"])
    with m2:
        ber_val = current_res["ber"]
        st.metric(
            "Bit Error Rate (BER)",
            f"{ber_val:.2e}" if ber_val > 0 else "0.00e+00",
            delta=f"{'Meets Target' if current_res['target_ber_satisfied'] else 'Target Violated'}",
            delta_color="normal" if current_res["target_ber_satisfied"] else "inverse",
        )
    with m3:
        st.metric("Net Goodput", f"{current_res['net_throughput_mbps']:.2f} Mbps")
    with m4:
        st.metric("Spectral Efficiency", f"{current_res['spectral_efficiency']:.2f} bps/Hz")
    with m5:
        st.metric("RMS EVM", f"{current_res['evm_rms']*100:.2f}%")

    # Quick Diagnosis Box
    st.markdown("---")
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("### 📋 Transmission Telemetry")
        telemetry_df = pd.DataFrame([
            {"Parameter": "Requested Mode", "Value": str(current_res["requested_mode"])},
            {"Parameter": "Active Modulation", "Value": str(current_res["mod_type"])},
            {"Parameter": "Channel SNR", "Value": f"{current_res['snr_db']:.1f} dB"},
            {"Parameter": "Propagation Model", "Value": str(current_res["channel_type"])},
            {"Parameter": "Transmitted Bits", "Value": f"{current_res['num_bits']:,}"},
            {"Parameter": "Observed Bit Errors", "Value": f"{current_res['bit_errors']:,}"},
            {"Parameter": "Target BER Requirement", "Value": f"{current_res['target_ber']:.1e}"},
            {"Parameter": "QoS Constraint Satisfied", "Value": "✅ YES" if current_res["target_ber_satisfied"] else "❌ NO"},
        ])
        st.dataframe(telemetry_df, hide_index=True, use_container_width=True)

    with col_d2:
        st.markdown("### 🔍 Channel State Measurements")
        feats = current_res["channel_features"]
        ch_df = pd.DataFrame([
            {"Metric": "Channel Mean Gain E[|H|]", "Value": f"{feats.get('h_mean', 1.0):.4f}"},
            {"Metric": "Frequency Selectivity Var(|H|)", "Value": f"{feats.get('h_var', 0.0):.4f}"},
            {"Metric": "Deep Fade Minimum Min(|H|)", "Value": f"{feats.get('h_min', 1.0):.4f}"},
            {"Metric": "Pilot Estimation Error Var", "Value": f"{feats.get('pilot_error_var', 0.0):.4e}"},
            {"Metric": "Raw PHY Data Rate", "Value": f"{current_res['raw_bitrate_mbps']:.2f} Mbps"},
            {"Metric": "Processing Execution Latency", "Value": f"{current_res['execution_time_sec']*1000:.2f} ms"},
        ])
        st.dataframe(ch_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: SIGNAL VISUALIZATION (Section D)
# ---------------------------------------------------------
with tab_signals:
    st.subheader("📊 Signal Domain & Constellation Analysis")

    col_sig1, col_sig2 = st.columns([1, 1])
    with col_sig1:
        fig_const = plot_constellation(
            tx_symbols=current_res["tx_symbols_sample"],
            rx_symbols=current_res["rx_symbols_sample"],
            mod_type=current_res["mod_type"],
            evm_rms=current_res["evm_rms"],
        )
        st.plotly_chart(fig_const, use_container_width=True)

    with col_sig2:
        fig_cfr = plot_channel_frequency_response(
            true_cfr=current_res["h_true_cfr"],
            est_cfr=current_res["h_est_cfr"],
            pilot_idx=[11, 25, 39, 53],
        )
        st.plotly_chart(fig_cfr, use_container_width=True)

    st.markdown("---")
    st.subheader("Time-Domain Baseband Waveform (with Cyclic Prefix Insertion)")
    fig_wave = plot_ofdm_waveform(
        tx_time_signal=current_res["tx_time_signal_sample"],
        rx_time_signal=current_res["rx_time_signal_sample"],
    )
    st.plotly_chart(fig_wave, use_container_width=True)

    # Example Transmitted and Recovered Bit Stream Sample
    st.markdown("#### Sample Bit Stream Comparison (First 32 Transmitted Bits)")
    tx_sample = current_res["tx_bits_sample"]
    rx_sample = current_res["rx_bits_sample"]

    bit_display_items = []
    for i in range(min(32, len(tx_sample))):
        match = (tx_sample[i] == rx_sample[i])
        bit_display_items.append({
            "Bit Index": i,
            "Tx Bit": tx_sample[i],
            "Rx Bit": rx_sample[i],
            "Status": "✅ Match" if match else "❌ Error",
        })
    st.dataframe(pd.DataFrame(bit_display_items).T, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: PERFORMANCE ANALYSIS (Section E & Scientific Validation)
# ---------------------------------------------------------
with tab_analysis:
    st.subheader("📈 Scientific Performance Benchmarks")
    st.markdown(
        "Comparative evaluation across **Fixed Modulation (BPSK, QPSK, 16-QAM, 64-QAM)**, "
        "**Threshold-Based Adaptation**, and **Machine-Learning Adaptation**."
    )

    bench_csv = os.path.join(os.getcwd(), "results", "benchmark_results.csv")
    if os.path.exists(bench_csv):
        df_bench = pd.read_csv(bench_csv)
    else:
        st.info("Generating baseline benchmark dataset...")
        snr_sweep = np.arange(0, 26, 2)
        schemes = ["BPSK", "QPSK", "16-QAM", "64-QAM", "Adaptive-Threshold", "Adaptive-AI"]
        df_bench = sim_iface.run_snr_sweep(
            snr_range=snr_sweep,
            mod_schemes=schemes,
            channel_type=ch_type,
            target_ber=target_ber,
            ai_engine=ai_engine,
        )

    col_perf1, col_perf2 = st.columns([1, 1])
    with col_perf1:
        fig_ber = plot_ber_curves(
            df_bench,
            target_ber=target_ber,
            channel_type=ch_type,
            show_theoretical=True,
        )
        st.plotly_chart(fig_ber, use_container_width=True)

    with col_perf2:
        fig_thru = plot_throughput_curves(df_bench)
        st.plotly_chart(fig_thru, use_container_width=True)

    st.markdown("### 📊 Benchmark Summary Table")
    st.dataframe(df_bench, use_container_width=True, height=280)

# ---------------------------------------------------------
# TAB 5: AI DECISION EXPLANATION (Section F)
# ---------------------------------------------------------
with tab_ai:
    st.subheader("🤖 AI Adaptation Engine Reasoning & Guardrails")

    ai_info = current_res.get("ai_decision_info")
    if ai_info is None:
        st.warning("Current simulation did not invoke the AI engine (fixed modulation selected). Switch to 'Adaptive-AI' mode in the sidebar to view live AI reasoning.")
    else:
        col_ai1, col_ai2 = st.columns([1, 1])
        with col_ai1:
            st.markdown(
                f"""
                <div class="card-box">
                  <h3>Modulation Decision: <span style="color:#00e676;">{ai_info['selected_mod']}</span></h3>
                  <p><b>Raw AI Recommendation:</b> {ai_info.get('raw_ai_recommendation', 'N/A')}</p>
                  <p><b>Model Confidence:</b> {ai_info.get('confidence', 1.0)*100:.1f}%</p>
                  <p><b>Fallback Status:</b> {'⚠️ ACTIVATED' if ai_info.get('fallback_triggered') else '✅ INACTIVE (Normal)'}</p>
                  <p><b>Decision Reason:</b> {ai_info.get('fallback_reason', 'Optimum throughput under BER constraint.')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Features Table
            st.markdown("#### Input Feature Vector")
            feats_dict = ai_info.get("feature_vector", {})
            st.json(feats_dict)

        with col_ai2:
            probs = ai_info.get("probabilities")
            if probs:
                fig_prob = plot_ai_probability_bar(probs, ai_info["selected_mod"])
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.info("Class probabilities available when AI model is active.")

            st.markdown(
                """
                <div class="card-box">
                <h4>🛡️ Safety Guardrail Logic</h4>
                <ol>
                  <li><b>Estimation Error Check:</b> If pilot error variance > 0.20, force step down by 1 modulation rank.</li>
                  <li><b>Confidence Check:</b> If max class probability < 55%, demote to next safer modulation.</li>
                  <li><b>Deep Null Guardrail:</b> If min |H| < 0.15, cap modulation at QPSK to prevent link disconnect.</li>
                </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# TAB 6: EXPERIMENT REPORTS (Section G)
# ---------------------------------------------------------
with tab_reports:
    st.subheader("📄 Experiment Reports & Export")

    col_rep1, col_rep2 = st.columns([1, 1])
    with col_rep1:
        st.markdown("#### 📥 Download Simulation Data")
        if os.path.exists(bench_csv):
            with open(bench_csv, "r") as f:
                csv_data = f.read()
            st.download_button(
                label="📥 Download Benchmark Results (CSV)",
                data=csv_data,
                file_name="ofdm_simulation_benchmark_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # JSON Configuration Export
        current_cfg = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mod_mode": mod_mode,
            "snr_db": snr_db,
            "channel_type": ch_type,
            "num_taps": num_taps,
            "target_ber": target_ber,
            "num_bits": num_bits,
            "outcomes": {
                "active_mod": current_res["mod_type"],
                "ber": current_res["ber"],
                "net_throughput_mbps": current_res["net_throughput_mbps"],
                "evm_rms": current_res["evm_rms"],
            },
        }
        st.download_button(
            label="📥 Export Current Simulation Config (JSON)",
            data=json.dumps(current_cfg, indent=2),
            file_name="simulation_run_config.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_rep2:
        st.markdown(
            """
            <div class="card-box">
            <h4>📋 Academic Experiment Metadata</h4>
            <ul>
              <li><b>Simulation Seed:</b> {}</li>
              <li><b>FFT Normalization:</b> Unitary 1/sqrt(N_fft)</li>
              <li><b>Overhead Factor:</b> (48 data subcarriers / 80 total samples) = 0.60</li>
              <li><b>Finite Bit Resolution:</b> BER < 1/N_bits indicates zero errors observed</li>
              <li><b>Reproducibility:</b> Deterministic pseudo-random sequence generator</li>
            </ul>
            </div>
            """.format(rand_seed),
            unsafe_allow_html=True,
        )
