%% RUN_SIMULATION.M
% =========================================================================
% Master Monte Carlo Simulation Script for Adaptive OFDM Communication
% =========================================================================
% Executes comprehensive BER vs SNR and Throughput evaluations for:
% 1. Fixed BPSK
% 2. Fixed QPSK
% 3. Fixed 16-QAM
% 4. Fixed 64-QAM
% 5. Adaptive Threshold-Based Modulation
%
% Saves results to:
% - ../results/matlab_simulation_results.csv
% - ../results/matlab_simulation_results.mat
% =========================================================================

clear; clc; close all;

% 1. Initialize parameters
initialize_simulation;

snr_range = simParams.snr_sweep_dB;
schemes = {'BPSK', 'QPSK', '16-QAM', '64-QAM', 'Adaptive'};
num_snr = length(snr_range);
num_schemes = length(schemes);

% Preallocate metric matrices
ber_matrix        = zeros(num_schemes, num_snr);
throughput_matrix = zeros(num_schemes, num_snr);
evm_matrix        = zeros(num_schemes, num_snr);

fprintf('\nStarting Monte Carlo Simulations across SNR grid...\n');
fprintf('Channel Model: %s (Multipath Taps: %d)\n', simParams.default_channel, simParams.num_taps);
fprintf('Total Transmitted Bits per point: %d\n', simParams.num_bits);
fprintf('-----------------------------------------------------\n');
fprintf(' SNR(dB) |   BPSK   |   QPSK   |  16-QAM  |  64-QAM  | Adaptive \n');
fprintf('-----------------------------------------------------\n');

% Threshold switching boundaries for target BER = 1e-2
snr_th_bpsk_qpsk  = 6.0;
snr_th_qpsk_16qam = 14.0;
snr_th_16qam_64qam= 22.0;

% Monte Carlo Loop
for s_idx = 1:num_snr
    snr_val = snr_range(s_idx);
    row_ber = zeros(1, num_schemes);

    for m_idx = 1:num_schemes
        scheme = schemes{m_idx};

        % Determine modulation to use
        if strcmp(scheme, 'Adaptive')
            if snr_val < snr_th_bpsk_qpsk
                active_mod = 'BPSK';
            elseif snr_val < snr_th_qpsk_16qam
                active_mod = 'QPSK';
            elseif snr_val < snr_th_16qam_64qam
                active_mod = '16-QAM';
            else
                active_mod = '64-QAM';
            end
        else
            active_mod = scheme;
        end

        bps = simParams.bps_map(active_mod);

        % Generate random binary source bits
        rng(42 + s_idx * 10 + m_idx);
        tx_bits = randi([0, 1], simParams.num_bits, 1);

        % OFDM Modulation (Transmitter)
        [tx_signal, tx_syms, num_syms, pad_len] = ofdm_modulate(tx_bits, active_mod, simParams);

        % Wireless Propagation Channel
        [rx_signal, h_cir, h_cfr, noise_var] = wireless_channel(...
            tx_signal, snr_val, simParams.default_channel, simParams);

        % OFDM Demodulation & Equalization (Receiver)
        [rx_bits, eq_syms, ber, evm, ch_feat] = ofdm_demodulate(...
            rx_signal, tx_bits, active_mod, num_syms, pad_len, noise_var, simParams);

        % Effective Net Throughput (Mbps)
        % Bandwidth = 20 MHz, duration = 4 us per OFDM symbol
        symbol_duration = simParams.symbol_len / simParams.bandwidth_Hz;
        raw_bitrate_mbps = (simParams.N_data * bps) / (symbol_duration * 1e6);
        net_throughput_mbps = raw_bitrate_mbps * (1 - ber);

        ber_matrix(m_idx, s_idx) = ber;
        throughput_matrix(m_idx, s_idx) = net_throughput_mbps;
        evm_matrix(m_idx, s_idx) = evm;
        row_ber(m_idx) = ber;
    end

    fprintf('  %5.1f  |  %7.4f |  %7.4f |  %7.4f |  %7.4f |  %7.4f\n', ...
        snr_val, row_ber(1), row_ber(2), row_ber(3), row_ber(4), row_ber(5));
end

fprintf('-----------------------------------------------------\n');
fprintf('Simulations Completed Successfully!\n');

%% 2. Save Simulation Results
results_dir = fullfile('..', 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

mat_file = fullfile(results_dir, 'matlab_simulation_results.mat');
save(mat_file, 'snr_range', 'schemes', 'ber_matrix', 'throughput_matrix', 'evm_matrix', 'simParams');
fprintf('Results saved to: %s\n', mat_file);

% Export to CSV for Python/Dashboard consumption
csv_file = fullfile(results_dir, 'matlab_simulation_results.csv');
fid = fopen(csv_file, 'w');
fprintf(fid, 'snr_db,scheme,ber,net_throughput_mbps,evm_rms\n');
for s_idx = 1:num_snr
    for m_idx = 1:num_schemes
        fprintf(fid, '%.2f,%s,%.6e,%.4f,%.4f\n', ...
            snr_range(s_idx), schemes{m_idx}, ...
            ber_matrix(m_idx, s_idx), ...
            throughput_matrix(m_idx, s_idx), ...
            evm_matrix(m_idx, s_idx));
    end
end
fclose(fid);
fprintf('CSV Export saved to: %s\n', csv_file);

%% 3. Plot Results
plot_results;
