# AI-Driven Adaptive OFDM Digital Communication System
### Using MATLAB/Simulink and Python

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![Pytest Tests](https://img.shields.io/badge/Tests-18%2F18%20Passed-brightgreen.svg)]()
[![Hardware Free](https://img.shields.io/badge/Hardware-100%25%20Software--Only-success.svg)]()

A complete, technically accurate, software-only Digital Communication engineering project designed for a three-member engineering team. The system models an IEEE 802.11a/g-style OFDM communication chain in both **MATLAB/Simulink** and **Python**, coupled with an autonomous **Machine Learning Link Adaptation Engine** and an interactive **Streamlit Engineering Dashboard**.

---

## 1. Project Highlights

- **Standard OFDM PHY Chain:** 64-point FFT/IFFT, 52 active subcarriers (48 data, 4 pilots at standard indices `[-21, -7, 7, 21]`), 16-sample cyclic prefix (CP), and DC/guard nulls.
- **Multi-Scheme Digital Modulation:** BPSK, QPSK, 16-QAM, and 64-QAM with Gray constellation mapping and unit average symbol power normalization ($\mathbb{E}[|s|^2] = 1$).
- **Realistic Wireless Channels:** Calibrated complex AWGN, multipath Rayleigh fading with exponential Power Delay Profile (PDP), and configurable Carrier Frequency Offset (CFO).
- **Pilot-Aided Channel Estimation & Equalization:** Least Squares (LS) pilot estimation, 1D linear channel frequency response interpolation, and Zero-Forcing (ZF) / MMSE equalization.
- **Machine Learning Adaptation Engine:** Random Forest classifier trained on 2,000 multi-channel Monte Carlo realizations (84.20% accuracy) optimizing goodput under strict QoS constraints ($\text{BER} \le \text{BER}_{target}$).
- **Autonomous Safety Guardrails:** Dynamic fallback triggers on high channel estimation error ($\sigma_e^2 > 0.20$), low classifier confidence ($< 55\%$), and deep multipath notches ($\min |H| < 0.15$).
- **MATLAB & Simulink Integration:** Dual-engine architecture with MATLAB scripts (`initialize_simulation.m`, `run_simulation.m`, `generate_dataset.m`, `plot_results.m`), programmatic Simulink model generator (`build_simulink_model.m`), and JSON/CSV data exchange bridge.
- **100% Software-Only:** Runs entirely on standard computers without SDRs, RF hardware, or paid cloud services.

---

## 2. Directory Structure

```
D:\DC project\
├── app.py                      # Interactive Streamlit Web Laboratory Dashboard
├── train_ai_model.py           # Dataset generator, model training, and benchmark script
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview & running instructions
├── config/
│   └── default_config.json     # Default simulation and AI configurations
├── python_src/
│   ├── __init__.py
│   ├── modulation.py           # Constellation mapping, demapping, Gray coding, theoretical BER
│   ├── ofdm_sim.py             # OFDM transceiver, pilot insertion, IFFT, CP, channel, receiver, equalizer
│   ├── channel.py              # AWGN and Rayleigh multipath channel models
│   ├── ai_model.py             # Feature extraction, ML adaptation, threshold baseline, fallback engine
│   ├── simulation_interface.py # MATLAB engine bridge + JSON/CSV data exchange + local execution
│   ├── metrics.py              # BER, SER, EVM, Throughput, Spectral Efficiency, Shannon Capacity
│   └── visualization.py        # Interactive Plotly chart generators
├── matlab/
│   ├── initialize_simulation.m # MATLAB parameter setup & constellation definitions
│   ├── run_simulation.m        # Standalone MATLAB Monte Carlo simulation script
│   ├── ofdm_modulate.m         # MATLAB OFDM transmitter function
│   ├── ofdm_demodulate.m       # MATLAB OFDM receiver function
│   ├── wireless_channel.m      # MATLAB AWGN & Rayleigh multipath channel function
│   ├── generate_dataset.m      # MATLAB script to export AI training dataset to CSV
│   ├── plot_results.m          # MATLAB plotting script for BER, throughput, constellations
│   ├── build_simulink_model.m  # Programmatic Simulink model generator script
│   └── simulink_guide.md       # Block diagram, subsystem layout, and manual construction steps
├── data/
│   └── simulation_dataset.csv  # 2,000 multi-channel records for AI training
├── models/
│   ├── adaptive_rf_model.joblib # Trained Random Forest model for adaptive modulation
│   └── model_metadata.json     # Feature list, classes, target BER, training metrics
├── results/
│   ├── benchmark_results.csv   # Pre-computed Monte Carlo benchmark curves
│   └── matlab_simulation_results.csv
├── tests/
│   ├── __init__.py
│   ├── test_modulation.py      # Unit tests for BPSK, QPSK, 16-QAM, 64-QAM mapping & BER
│   ├── test_ofdm.py            # Unit tests for OFDM end-to-end chain & channel equalization
│   ├── test_ai_model.py        # Unit tests for ML adaptation, threshold baseline, fallback safety
│   └── test_interface.py       # Unit tests for simulation interface & JSON/CSV bridge
└── docs/
    ├── academic_report.md      # Full Academic Project Report (Math, Literature, Results)
    ├── team_contributions.md   # 3-Member Engineering Team division and integration plan
    ├── viva_questions.md       # 20+ In-depth Viva Voce questions & answers
    └── presentation_outline.md # Slide-by-slide project presentation & demonstration outline
```

---

## 3. Installation & Setup

### Prerequisites
- **Python:** Python 3.12 (installed and verified).
- **MATLAB:** MATLAB R2022b / R2023a / R2024a (optional, for running MATLAB/Simulink scripts).

### Python Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run the Project

### 1. Launch the Interactive Dashboard
Run the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

The dashboard includes 6 tabs:
1. **System Overview:** Architecture diagram, parameter profile, and OFDM equations.
2. **Live Simulation:** Real-time transceiver execution with dynamic SNR, modulation selection, and KPI cards.
3. **Signal Visualization:** Interactive Plotly constellation diagrams (with EVM), channel frequency response, and time-domain waveforms.
4. **Performance Curves:** Comparative BER and throughput curves (Fixed vs. Adaptive vs. Threshold).
5. **AI Adaptation Engine:** Live decision reasoning card, feature inspection, probability distribution, and guardrail status.
6. **Experiment Reports:** One-click CSV and JSON simulation configuration export.

### 2. Execute Automated Unit Tests
Run the test suite:
```bash
pytest tests -v
```
*Outcome: 18 passed in ~2.7 seconds.*

### 3. Re-train AI Model & Generate Benchmark Sweep
```bash
python train_ai_model.py
```

### 4. Run MATLAB / Simulink Scripts
In MATLAB:
```matlab
cd 'D:\DC project\matlab'
initialize_simulation;
run_simulation;
```
To programmatically generate the Simulink model (`adaptive_ofdm_system.slx`):
```matlab
build_simulink_model;
```

---

## 5. Three-Member Team Responsibilities

| Member | Primary Domain | Modules |
| :--- | :--- | :--- |
| **Member 1** | Digital Communication & MATLAB/Simulink | `ofdm_modulate.m`, `wireless_channel.m`, `ofdm_demodulate.m`, `build_simulink_model.m`, `simulink_guide.md` |
| **Member 2** | Machine Learning & Data Engineering | `ai_model.py`, `train_ai_model.py`, `generate_dataset.m`, optimization objective, guardrails |
| **Member 3** | Integration & Software Development | `app.py`, `simulation_interface.py`, `visualization.py`, `metrics.py`, `tests/` |

See [docs/team_contributions.md](docs/team_contributions.md) for the complete integration plan.

---

## 6. Academic Deliverables & Documentation

- [Academic Project Report](docs/academic_report.md)
- [Viva Voce Questions & Answers](docs/viva_questions.md)
- [Presentation Outline](docs/presentation_outline.md)
- [Simulink Construction Guide](matlab/simulink_guide.md)
- [Team Contribution Matrix](docs/team_contributions.md)
