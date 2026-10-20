%% GENERATE_DATASET.M
% =========================================================================
% AI Training Dataset Generator for Adaptive OFDM Modulation
% =========================================================================
% Generates realistic channel state observations across diverse SNR and
% multipath fading conditions, and computes the ground-truth optimal
% modulation scheme (BPSK, QPSK, 16-QAM, 64-QAM) that maximizes throughput
% while satisfying a target BER constraint.
%
% Output:
%   ../data/matlab_ai_training_dataset.csv
% =========================================================================

clear; clc;
initialize_simulation;

num_samples = 1200;
target_ber = simParams.target_ber;

fprintf('Generating %d channel simulation training samples...\n', num_samples);

% Ensure data directory exists
data_dir = fullfile('..', 'data');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
csv_path = fullfile(data_dir, 'matlab_ai_training_dataset.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'snr_db,h_mean,h_var,h_min,pilot_error_var,target_ber_log10,ber_bpsk,ber_qpsk,ber_16qam,ber_64qam,optimal_mod_idx,optimal_mod_name\n');

mod_names = {'BPSK', 'QPSK', '16-QAM', '64-QAM'};
bps_vals  = [1, 2, 4, 6];

for i = 1:num_samples
    % Random SNR in [-2, 28] dB
    snr_val = -2.0 + 30.0 * rand();

    % Random channel configuration
    is_rayleigh = (rand() > 0.25);
    if is_rayleigh
        ch_type = 'Rayleigh';
        simParams.num_taps = randi([1, 6]);
        simParams.tap_delays = 0 : (simParams.num_taps - 1);
        simParams.decay_factor = 0.3 + 0.5 * rand();
    else
        ch_type = 'AWGN';
        simParams.num_taps = 1;
        simParams.tap_delays = 0;
    end

    % Send QPSK probe signal to estimate channel features
    probe_bits = randi([0, 1], 192, 1);
    [tx_probe, ~, n_sym, pad_len] = ofdm_modulate(probe_bits, 'QPSK', simParams);
    [rx_probe, ~, ~, noise_var]   = wireless_channel(tx_probe, snr_val, ch_type, simParams);
    [~, ~, ~, ~, ch_feat]         = ofdm_demodulate(rx_probe, probe_bits, 'QPSK', n_sym, pad_len, noise_var, simParams);

    % Effective SNR approximation for subcarrier channel
    eff_snr_lin = (10^(snr_val / 10)) * (ch_feat.h_mean^2) / (1.0 + 3.0 * ch_feat.h_var);
    eff_snr_db  = 10 * log10(max(eff_snr_lin, 1e-4));

    % Evaluate theoretical BER for all 4 modulations
    % Closed-form formulas matching standards
    ber_est = zeros(1, 4);
    for m = 1:4
        mod_name = mod_names{m};
        bps = bps_vals(m);
        eb_n0 = (10^(eff_snr_db / 10)) / bps;

        if strcmpi(ch_type, 'AWGN')
            switch mod_name
                case {'BPSK', 'QPSK'}
                    ber_est(m) = 0.5 * erfc(sqrt(eb_n0));
                case '16-QAM'
                    ber_est(m) = (3/8) * 0.5 * erfc(sqrt(0.4 * eb_n0));
                case '64-QAM'
                    ber_est(m) = (7/24) * 0.5 * erfc(sqrt((6/21) * eb_n0));
            end
        else
            switch mod_name
                case {'BPSK', 'QPSK'}
                    ber_est(m) = 0.5 * (1 - sqrt(eb_n0 / (1 + eb_n0)));
                case '16-QAM'
                    c = 0.4 * eb_n0;
                    ber_est(m) = (3/8) * (1 - sqrt(c / (1 + c)));
                case '64-QAM'
                    c = (1/7) * eb_n0;
                    ber_est(m) = (7/24) * (1 - sqrt(c / (1 + c)));
            end
        end
    end

    % Decision Rule: Select highest modulation that satisfies target BER
    best_idx = 0; % 0: BPSK, 1: QPSK, 2: 16QAM, 3: 64QAM
    max_rate = -1.0;

    for m = 1:4
        if ber_est(m) <= target_ber * 1.5
            rate = bps_vals(m) * (1 - ber_est(m));
            if rate > max_rate
                max_rate = rate;
                best_idx = m - 1;
            end
        end
    end

    fprintf(fid, '%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%.6e,%.6e,%.6e,%.6e,%d,%s\n', ...
        snr_val, ch_feat.h_mean, ch_feat.h_var, ch_feat.h_min, ...
        ch_feat.pilot_error_var, log10(target_ber), ...
        ber_est(1), ber_est(2), ber_est(3), ber_est(4), ...
        best_idx, mod_names{best_idx + 1});

    if mod(i, 200) == 0
        fprintf('  Generated %d / %d samples...\n', i, num_samples);
    end
end

fclose(fid);
fprintf('Dataset successfully written to: %s\n', csv_path);
