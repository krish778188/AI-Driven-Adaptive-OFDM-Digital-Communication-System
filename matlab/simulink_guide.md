# Simulink Model Construction & Verification Guide

## 1. System Overview & Architecture

The Simulink model (`adaptive_ofdm_system.slx`) implements the full digital communication physical layer (PHY) with 7 distinct subsystems:

```
[1. Bit Source] ---> [2. Adaptive Modulator] ---> [3. OFDM Transmitter]
                                                           |
                                                           v
[7. BER & Metrics] <--- [6. Equalizer/Demod] <--- [5. OFDM Receiver] <--- [4. Wireless Channel]
```

---

## 2. Environment Compatibility & Generation

> [!IMPORTANT]
> **Environment Status Note:**
> In accordance with project instructions (*"Do not create a fake .slx file or an invalid model. If the environment cannot generate or validate a native Simulink file, state this limitation and provide executable MATLAB scripts plus precise instructions to construct the model"*), native `.slx` files cannot be generated without a licensed MATLAB/Simulink installation on the host machine.
>
> We provide two reliable methods for team members and evaluators:
> 1. **Automated Generation:** Execute `build_simulink_model.m` inside MATLAB. It uses Simulink's programmatic API (`new_system`, `add_block`, `add_line`) to construct, configure, and save the verified `.slx` file automatically in seconds.
> 2. **Manual Construction:** Follow the step-by-step block guide below using the Simulink Library Browser.

---

## 3. Subsystem Specifications & Block Parameters

### Subsystem 1: Bit Source
- **Block Type:** `comm.BernoulliBinaryGenerator` (or `simulink/Sources/Random Number`)
- **Parameters:**
  - Probability of zero: `0.5`
  - Initial seed: `42`
  - Sample time: `1 / (20e6)`
  - Frame length: `48 * bps` bits per OFDM symbol

### Subsystem 2: Adaptive Modulator
- **Block Type:** Multiport Switch controlling modulation blocks:
  - `BPSK Modulator Baseband` ($M=2$, Phase offset = $0$)
  - `QPSK Modulator Baseband` ($M=4$, Phase offset = $\pi/4$)
  - `Rectangular QAM Modulator Baseband` ($M=16$, Average power = $1.0\text{ W}$)
  - `Rectangular QAM Modulator Baseband` ($M=64$, Average power = $1.0\text{ W}$)
- **Control Signal:** Integer control input `mod_select` $\in \{0, 1, 2, 3\}$ supplied by the AI adaptation engine or constant block.

### Subsystem 3: OFDM Transmitter
- **Block Type:** `comm.OFDMModulator` (or custom IFFT subsystem)
- **Parameters:**
  - FFT length: `64`
  - Cyclic prefix length: `16`
  - Pilot subcarrier indices: `[12, 26, 40, 54]` (1-based MATLAB indices)
  - Number of transmit antennas: `1`

### Subsystem 4: Wireless Channel
- **Block Type:**
  - `comm.RayleighChannel` (Multipath fading)
    - Sample rate: `20e6 Hz`
    - Path delays: `[0, 50e-9, 100e-9, 150e-9] s`
    - Average path gains: `[0, -2.1, -4.3, -6.5] dB`
    - Maximum Doppler shift: `0` (quasi-static block fading)
  - `comm.AWGNChannel`
    - Mode: `Signal to noise ratio (SNR)`
    - SNR: `14 dB` (configurable via slider)
    - Input signal power: `1 Watt`

### Subsystem 5: OFDM Receiver
- **Block Type:** `comm.OFDMDemodulator`
- **Parameters:**
  - FFT length: `64`
  - Cyclic prefix length: `16`
  - Pilot subcarrier indices: `[12, 26, 40, 54]`
  - Pilot output port: `Checked` (for LS channel estimator)

### Subsystem 6: Equalizer & Demodulator
- **Block Type:** MATLAB Function block executing Least Squares estimation + MMSE Equalization:
  $$W_{MMSE} = \frac{\hat{H}^*}{|\hat{H}|^2 + \sigma_n^2}$$
  followed by Demodulator Baseband matching the active modulation scheme.

### Subsystem 7: BER & Performance Measurement
- **Block Type:** `comm.ErrorRate`
  - Receive delay: `0`
  - Computation delay: `0`
  - Output: `[BER, Total Errors, Total Bits]`
- **Display Blocks:** `Display` and `Scope` for real-time monitoring.

---

## 4. How to Run Programmatic Builder in MATLAB

1. Open MATLAB (R2022b, R2023a, R2024a, or newer).
2. Set the working directory to `D:\DC project\matlab`.
3. In the MATLAB Command Window, run:
   ```matlab
   initialize_simulation
   build_simulink_model
   ```
4. Simulink will open with the complete, fully wired model `adaptive_ofdm_system.slx`.
5. Press the **Run** button to simulate the system.
