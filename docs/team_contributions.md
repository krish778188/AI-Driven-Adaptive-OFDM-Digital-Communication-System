# Three-Member Engineering Team Division & Integration Plan

## Project: AI-Driven Adaptive OFDM Digital Communication System Using MATLAB/Simulink and Python

---

## 1. Team Role Allocation Matrix

| Team Member | Primary Domain | Core Responsibilities & Modules | Viva Focus Areas |
| :--- | :--- | :--- | :--- |
| **Member 1** | Digital Communication & MATLAB/Simulink | - MATLAB OFDM transmission chain (`ofdm_modulate.m`)<br>- Multipath Rayleigh & AWGN channel model (`wireless_channel.m`)<br>- Pilot-aided LS channel estimation & MMSE/ZF equalization (`ofdm_demodulate.m`)<br>- Programmatic Simulink model builder (`build_simulink_model.m`) & guide (`simulink_guide.md`) | - Orthogonality, IFFT/FFT scaling<br>- CP duration vs delay spread<br>- Pilot placement & interpolation<br>- Zero-Forcing vs MMSE derivation |
| **Member 2** | Machine Learning & Data Engineering | - Dataset generation pipeline (`generate_dataset.m` & `ai_model.py`)<br>- Adaptive modulation objective function formulation<br>- Scikit-learn Random Forest / Gradient Boosting model training & tuning<br>- Autonomous fallback safety guardrails & feature engineering | - Objective function & target BER constraint<br>- Feature importance ($E[\|H\|]$, $\text{Var}(\|H\|)$, $\sigma_e^2$)<br>- Overfitting prevention & cross-validation<br>- Guardrail threshold selection |
| **Member 3** | Full-Stack Integration & Software Engineering | - MATLAB–Python simulation interface (`simulation_interface.py`)<br>- Interactive Streamlit engineering dashboard (`app.py`)<br>- Plotly signal & performance visualizers (`visualization.py`)<br>- Metric evaluators & automated pytest suite (`test_*.py`) | - Bridge architecture (Live Engine vs JSON/CSV)<br>- Scientific metric definitions (EVM, BER, Throughput)<br>- Test case design & verification<br>- Deployment, reproducibility & demo workflow |

---

## 2. Detailed Member Contributions

### Member 1: MATLAB/Simulink Communication Engineer
- **Detailed Tasks:**
  1. Formulated the discrete-time baseband OFDM model using standard IEEE 802.11a/g parameter specifications ($N_{fft} = 64$, $N_{cp} = 16$, 52 active subcarriers, 4 pilot tones).
  2. Implemented Gray-coded constellation mappers and demappers for BPSK, QPSK, 16-QAM, and 64-QAM with unit average symbol power normalization ($\mathbb{E}[|s|^2] = 1$).
  3. Developed the multipath Rayleigh fading channel using a tapped-delay line with exponential power delay profile and calibrated complex AWGN generation.
  4. Constructed the receiver processing pipeline: cyclic prefix removal, FFT demodulation, Least Squares (LS) channel estimation at pilot subcarriers $[-21, -7, 7, 21]$, linear channel interpolation, and MMSE/ZF equalization.
  5. Authored `build_simulink_model.m` which programmatically builds `adaptive_ofdm_system.slx` via the Simulink API with 7 modular subsystems.

### Member 2: Machine Learning & Adaptation Specialist
- **Detailed Tasks:**
  1. Formulated the constrained optimization objective function:
     $$\max_{M \in \{2, 4, 16, 64\}} \left[ \log_2(M) \cdot (1 - \text{BER}(M, \mathbf{h})) \right] \quad \text{subject to } \text{BER} \le \text{BER}_{\text{target}}$$
  2. Engineered 6 key channel state features: estimated SNR ($\text{dB}$), mean channel gain $E[|H|]$, frequency selectivity variance $\text{Var}(|H|)$, deep-null indicator $\min(|H|)$, pilot estimation error variance $\sigma_e^2$, and logarithmic target BER $\log_{10}(\text{BER}_{\text{target}})$.
  3. Built the dataset generator producing 2,000 multi-channel samples across AWGN, flat Rayleigh, and frequency-selective Rayleigh fading.
  4. Trained and cross-validated the Random Forest classifier achieving 84.20% test accuracy, saving the serializable model artifact to `models/adaptive_rf_model.joblib`.
  5. Designed three autonomous fallback guardrails:
     - Pilot estimation error guardrail ($\sigma_e^2 > 0.20 \implies$ demote modulation).
     - Model confidence guardrail ($P_{max} < 55\% \implies$ demote modulation).
     - Deep null guardrail ($\min(|H|) < 0.15 \implies$ cap at QPSK).

### Member 3: Software Integration & UI Developer
- **Detailed Tasks:**
  1. Built the dual-mode integration bridge `simulation_interface.py` supporting both the live MATLAB Engine API (`matlab.engine`) and high-performance JSON/CSV file-based parameter/result exchange.
  2. Created the high-fidelity reference Python communication engine ensuring full execution capability in environments without a MATLAB license.
  3. Designed and deployed the Streamlit laboratory dashboard (`app.py`) featuring 6 comprehensive tabs: System Overview, Live Simulation, Signal Visualization, Performance Curves, AI Decision Explanation, and Experiment Reports.
  4. Implemented interactive Plotly visualizers: I/Q constellation diagram with EVM annotation, channel frequency response $|H(f)|$ with pilot markers, time-domain waveform with CP markers, and semi-log BER vs SNR curves with theoretical bounds.
  5. Authored the automated unit test suite (`tests/`) achieving 100% pass rate across 18 tests covering modulation roundtrip, OFDM dimensioning, equalizer modes, AI prediction, and JSON export.

---

## 3. Shared Integration & Team Cohesion Plan

To ensure all team members understand the entire system and speak with unified technical confidence during viva examinations:

1. **System Walkthrough Sessions:**
   - Every member must run both the Python dashboard (`streamlit run app.py`) and the MATLAB simulation scripts (`run_simulation.m`).
   - Member 1 explains the mathematical rationale behind cyclic prefix length ($T_{cp} > \tau_{max}$) and MMSE equalizer weighting to Members 2 and 3.
   - Member 2 explains the feature importance and why Random Forest outperforms static threshold switching in frequency-selective fading to Members 1 and 3.
   - Member 3 walks through the interface architecture, data contracts, and live demo steps.

2. **Integration Touchpoints:**
   - **Data Contract:** JSON configuration format defined in `config/default_config.json` is the single source of truth for simulation parameters.
   - **Feature Contract:** Exact feature naming convention (`snr_db`, `h_mean`, `h_var`, `h_min`, `pilot_error_var`, `target_ber_log10`) shared identically between MATLAB dataset generator and Python classifier.
   - **Metric Contract:** BER, RMS EVM, and Net Throughput formulas implemented with identical mathematical definitions across both MATLAB and Python.
