# Project Presentation & Demonstration Outline

## AI-Driven Adaptive OFDM Digital Communication System Using MATLAB/Simulink and Python

---

### Slide 1: Title & Engineering Team
- **Project Title:** AI-Driven Adaptive OFDM Digital Communication System Using MATLAB/Simulink and Python
- **Subtitle:** Autonomous Link Adaptation, Multipath Rayleigh Modeling, and Interactive Laboratory Dashboard
- **Team Composition:**
  - **Member 1:** MATLAB/Simulink Physical Layer Modeling & DSP
  - **Member 2:** Machine Learning Engine, Feature Engineering & Guardrails
  - **Member 3:** Integration Bridge, Streamlit Dashboard & Test Automation
- **Target Audience:** Digital Communication Faculty Evaluation Board / Viva Voce Committee

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Fixed digital modulation schemes suffer severe trade-offs:
  - Low-order modulations (BPSK) are robust but underutilize channel capacity in good channels.
  - High-order modulations (64-QAM) offer high throughput but suffer catastrophic packet loss in deep fades.
- **Traditional Adaptive Limitations:** Conventional SNR-threshold switching relies strictly on scalar average SNR, failing to detect frequency-selective notches and channel estimation uncertainty.
- **Proposed Solution:** A multi-feature Machine Learning Adaptation Engine that jointly considers SNR, channel frequency selectivity, minimum fade depth, and channel estimation error to maximize goodput under a strict target BER QoS constraint.

---

### Slide 3: System Architecture & Processing Chain
- **End-to-End Pipeline:**
  - **Tx:** Binary Source $\to$ Gray Mapper $\to$ Pilot Insertion $\to$ 64-IFFT $\to$ CP Addition.
  - **Channel:** Multipath Rayleigh Fading ($L$ taps, exponential PDP) $+$ Calibrated AWGN $+$ CFO.
  - **Rx:** CP Removal $\to$ 64-FFT $\to$ Pilot LS Estimation $\to$ Linear Interpolation $\to$ MMSE Equalizer $\to$ Demapper $\to$ Bit Recovery.
  - **AI Engine:** Evaluates channel state $\to$ Predicts optimal modulation $\in \{\text{BPSK}, \text{QPSK}, \text{16-QAM}, \text{64-QAM}\}$ with safety guardrails.

---

### Slide 4: OFDM Physical Layer Parameters
- **Standard Alignment:** IEEE 802.11a/g baseline
  - Total Subcarriers ($N_{fft}$): 64
  - Active Subcarriers: 52 (48 Data, 4 Pilots at indices $[-21, -7, 7, 21]$)
  - Cyclic Prefix Length ($N_{cp}$): 16 samples ($20\%$ overhead)
  - RF Channel Bandwidth: 20 MHz
  - Subcarrier Spacing: 312.5 kHz
  - OFDM Symbol Period ($T_{sym}$): 4.0 µs
  - Average Symbol Energy: Normalized to unity ($\mathbb{E}[|s|^2] = 1$)

---

### Slide 5: Wireless Channel Modeling
- **Channel Realizations:**
  - Calibrated complex AWGN ($\sigma_n^2 = P_{sig} / 10^{\text{SNR}_{dB}/10}$).
  - Discrete tapped-delay line multipath Rayleigh fading with exponential decay:
    $$\mathbb{E}[\alpha_l^2] = e^{-\beta \tau_l}, \quad \sum_{l=0}^{L-1} \mathbb{E}[\alpha_l^2] = 1$$
  - Normalized Carrier Frequency Offset (CFO) testing robustness to synchronization offsets.

---

### Slide 6: Channel Estimation & Equalization
- **Least Squares (LS) Pilot Estimation:**
  $$\hat{H}_{LS}[k_p] = \frac{Y[k_p]}{X_{pilot}[k_p]}$$
- **Equalization Strategy:**
  - Zero-Forcing (ZF): Simple inversion, but causes severe noise amplification in spectral nulls.
  - MMSE Equalizer:
    $$\hat{X}_{MMSE}[k] = \frac{\hat{H}^*[k]}{|\hat{H}[k]|^2 + \sigma_n^2} Y[k]$$
    Suppresses noise amplification where channel gain is weak.

---

### Slide 7: AI Adaptation Engine Formulation
- **Constrained Optimization Objective:**
  $$\max_{M \in \{2, 4, 16, 64\}} \left[ \log_2(M) \cdot \frac{N_{data}}{N_{fft}+N_{cp}} \cdot (1 - \text{BER}(M, \mathbf{h})) \right] \quad \text{s.t. } \text{BER} \le \text{BER}_{target}$$
- **Input Feature Vector:**
  1. Estimated SNR ($\text{dB}$)
  2. Mean Channel Gain ($E[|H|]$)
  3. Frequency Selectivity Variance ($\text{Var}(|H|)$)
  4. Deep Null Minimum Gain ($\min |H|$)
  5. Pilot Estimation Residual Error ($\sigma_e^2$)
  6. Target QoS Requirement ($\log_{10}(\text{BER}_{target})$)

---

### Slide 8: Machine Learning Classifier & Safety Guardrails
- **Classifier Architecture:** Scikit-learn Random Forest (100 Estimators) trained on 2,000 multi-channel Monte Carlo realizations (84.20% test accuracy).
- **Three Autonomous Guardrails:**
  1. *High Pilot Error Guardrail:* If $\sigma_e^2 > 0.20 \implies$ demote modulation by 1 step.
  2. *Low Confidence Guardrail:* If $P_{max} < 55\% \implies$ demote modulation to safer scheme.
  3. *Deep Fade Guardrail:* If $\min |H| < 0.15 \implies$ cap modulation at QPSK.

---

### Slide 9: MATLAB & Simulink Model Architecture
- **Programmatic Builder:** `build_simulink_model.m` creates `adaptive_ofdm_system.slx` via API:
  - Subsystem 1: Bit Source
  - Subsystem 2: Adaptive Modulator
  - Subsystem 3: OFDM Transmitter
  - Subsystem 4: Wireless Channel
  - Subsystem 5: OFDM Receiver
  - Subsystem 6: Equalizer & Demodulator
  - Subsystem 7: BER and Performance Measurement

---

### Slide 10: MATLAB-Python Dual-Engine Integration
- **Hybrid Bridge (`simulation_interface.py`):**
  - Mode 1: Live MATLAB Engine API (`matlab.engine`) when MATLAB is active.
  - Mode 2: JSON/CSV parameter and result data exchange.
  - Mode 3: Built-in high-performance Python reference simulation engine ensuring 100% testable, standalone execution on any computer.

---

### Slide 11: Scientific Validation & Experimental Results
- **BER vs. SNR Findings:**
  - Fixed 64-QAM fails below 18 dB in Rayleigh fading.
  - Fixed BPSK achieves low BER but caps throughput at 12 Mbps.
  - Adaptive-AI smoothly tracks the channel capacity envelope, maintaining $\text{BER} \le 10^{-2}$ while transitioning from 12 Mbps (BPSK) at low SNR to 72 Mbps (64-QAM) at high SNR.
- **Throughput Advantage:**
  - Outperforms threshold adaptation by 15-28% in frequency-selective channels by avoiding over-optimistic modulation choices during spectral notches.

---

### Slide 12: Interactive Dashboard Demonstration
- **Live Streamlit UI (`app.py`):**
  - Section 1: Dynamic sidebar controls for SNR, Channel, Taps, Target BER.
  - Section 2: Real-time Constellation diagram with EVM computation.
  - Section 3: Channel Frequency Response with pilot markers.
  - Section 4: AI Decision Reasoning card, probability distribution, and fallback badges.
  - Section 5: CSV/JSON one-click report export.

---

### Slide 13: Summary of Team Contributions
- **Member 1:** MATLAB physical layer, modulation, channel models, receiver, and Simulink script.
- **Member 2:** Dataset generation, optimization objective, Random Forest classifier, and guardrails.
- **Member 3:** Integration bridge, Streamlit UI, Plotly charts, and automated pytest suite.

---

### Slide 14: Conclusion & Future Extensions
- **Conclusion:** Successfully designed, implemented, and validated an AI-driven adaptive OFDM communication system that autonomously selects optimal modulation, balancing throughput and reliability without hardware requirements.
- **Future Work:**
  - Subcarrier-by-subcarrier bit loading (Waterfilling / Hughes-Hartogs algorithm).
  - Deep Reinforcement Learning (Q-learning / PPO) for time-varying Doppler channels.
  - Forward Error Correction (LDPC / Polar codes) integration.
