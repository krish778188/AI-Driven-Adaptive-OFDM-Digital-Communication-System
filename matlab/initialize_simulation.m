%% INITIALIZE_SIMULATION.M
% =========================================================================
% AI-Driven Adaptive OFDM Digital Communication System
% Global Parameter Initialization Script
% =========================================================================
% Defines all standard parameters matching IEEE 802.11a/g and LTE baseline:
% - Subcarrier allocations (FFT size, data, pilots, nulls)
% - Modulation alphabet tables (BPSK, QPSK, 16-QAM, 64-QAM) with unit energy
% - Channel parameters (AWGN, Rayleigh multipath delay taps, SNR grid)
% - Target BER requirements and simulation boundaries
% =========================================================================

clear; clc;

fprintf('=====================================================\n');
fprintf('  Initializing Adaptive OFDM Communication System    \n');
fprintf('=====================================================\n');

%% 1. OFDM System Architecture Parameters
simParams.N_fft         = 64;               % Total FFT/IFFT points
simParams.N_cp          = 16;               % Cyclic Prefix length (1/4 N_fft)
simParams.symbol_len    = simParams.N_fft + simParams.N_cp; % Total OFDM symbol length (80)
simParams.bandwidth_Hz  = 20e6;             % RF Channel Bandwidth (20 MHz)
simParams.subcarrier_spacing = simParams.bandwidth_Hz / simParams.N_fft; % 312.5 kHz
simParams.sampling_rate = simParams.bandwidth_Hz;
simParams.Ts            = 1 / simParams.sampling_rate; % Sample duration (50 ns)

%% 2. Subcarrier Allocation Grid (1-based MATLAB indices)
% Active 52 subcarriers in centered indices [-26..-1, 1..26]
% Convert to standard 1..64 FFT indices:
active_shifted = [-26:-1, 1:26];
pilots_shifted = [-21, -7, 7, 21];
data_shifted   = setdiff(active_shifted, pilots_shifted);

simParams.pilot_indices = sort(mod(pilots_shifted + simParams.N_fft, simParams.N_fft) + 1);
simParams.data_indices  = sort(mod(data_shifted + simParams.N_fft, simParams.N_fft) + 1);

simParams.N_pilots = length(simParams.pilot_indices); % 4 pilots
simParams.N_data   = length(simParams.data_indices);   % 48 data subcarriers

% Known deterministic BPSK pilot symbols (+1, -1, +1, +1)
simParams.pilot_values = [1.0; -1.0; 1.0; 1.0];

%% 3. Modulation Alphabets and Normalization Factors
% Unit average energy normalization: E_s = E{|s|^2} = 1
simParams.modulations = {'BPSK', 'QPSK', '16-QAM', '64-QAM'};
simParams.bps_map = containers.Map(...
    {'BPSK', 'QPSK', '16-QAM', '64-QAM'}, ...
    {1, 2, 4, 6});

% Energy normalization constants:
% BPSK: E = 1 -> norm = 1
% QPSK: E = 2 -> norm = 1/sqrt(2)
% 16-QAM: E = 10 -> norm = 1/sqrt(10)
% 64-QAM: E = 42 -> norm = 1/sqrt(42)
simParams.norm_factors = containers.Map(...
    {'BPSK', 'QPSK', '16-QAM', '64-QAM'}, ...
    {1.0, 1.0/sqrt(2.0), 1.0/sqrt(10.0), 1.0/sqrt(42.0)});

%% 4. Channel Simulation Specifications
simParams.channel_types = {'AWGN', 'Rayleigh'};
simParams.default_channel = 'Rayleigh';
simParams.num_taps      = 4;                % Number of multipath delay taps
simParams.tap_delays    = [0, 1, 2, 3];     % Discrete tap delays in samples
simParams.decay_factor  = 0.5;              % Exponential power decay factor
simParams.cfo_norm      = 0.0;              % Normalized CFO (Delta f / Delta f_sub)

%% 5. Simulation Sweep Grid
simParams.snr_sweep_dB  = 0:2:24;           % Sweep range from 0 to 24 dB
simParams.num_bits      = 20000;            % Bits per Monte Carlo trial
simParams.target_ber    = 1e-2;             % Quality of Service target BER
simParams.equalizer     = 'MMSE';           % 'ZF' or 'MMSE'

fprintf('System Parameters Successfully Configured:\n');
fprintf('  FFT Size:            %d\n', simParams.N_fft);
fprintf('  Cyclic Prefix:       %d samples\n', simParams.N_cp);
fprintf('  Data Subcarriers:    %d\n', simParams.N_data);
fprintf('  Pilot Subcarriers:   %d\n', simParams.N_pilots);
fprintf('  Bandwidth:           %.1f MHz\n', simParams.bandwidth_Hz / 1e6);
fprintf('  Target BER:          %.1e\n', simParams.target_ber);
fprintf('=====================================================\n');
