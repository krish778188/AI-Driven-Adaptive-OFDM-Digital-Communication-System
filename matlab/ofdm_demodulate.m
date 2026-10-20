function [rx_bits, eq_symbols, ber, evm_rms, ch_features] = ofdm_demodulate(rx_signal, tx_bits, mod_type, num_ofdm_syms, pad_len, noise_variance, simParams)
%% OFDM_DEMODULATE
% Implements the receiver chain of the OFDM communication system:
% 1. Serial-to-parallel conversion & Cyclic Prefix removal
% 2. Normalized FFT demodulation
% 3. Pilot symbol extraction & Least Squares (LS) channel estimation
% 4. 1D linear channel frequency response interpolation
% 5. Zero-Forcing (ZF) or MMSE equalization
% 6. Minimum Euclidean distance constellation demapping & Gray decoding
% 7. BER and EVM performance metric calculations
%
% Outputs:
%   rx_bits     - Recovered binary vector
%   eq_symbols  - Equalized complex symbols
%   ber         - Bit Error Rate
%   evm_rms     - RMS Error Vector Magnitude
%   ch_features - Extracted channel statistics for AI adaptation engine

    rx_signal = rx_signal(:);
    symbol_len = simParams.N_fft + simParams.N_cp;
    total_samples = num_ofdm_syms * symbol_len;

    % Align received sample length
    if length(rx_signal) < total_samples
        rx_signal = [rx_signal; zeros(total_samples - length(rx_signal), 1)];
    else
        rx_signal = rx_signal(1:total_samples);
    end

    % 1. Reshape into OFDM symbol blocks: (symbol_len x num_ofdm_syms)
    rx_blocks_cp = reshape(rx_signal, symbol_len, num_ofdm_syms);

    % 2. Remove Cyclic Prefix
    rx_blocks = rx_blocks_cp(simParams.N_cp + 1 : end, :);

    % 3. Normalized FFT Demodulation
    rx_freq_grid = fft(rx_blocks, simParams.N_fft, 1) / sqrt(simParams.N_fft);

    % 4. Pilot Extraction and LS Channel Estimation
    rx_pilots = rx_freq_grid(simParams.pilot_indices, :);
    pilot_ref = repmat(simParams.pilot_values, 1, num_ofdm_syms);
    h_hat_pilots = rx_pilots ./ pilot_ref; % Least Squares

    % Average pilot estimates across symbols for static/block fading
    avg_h_pilots = mean(h_hat_pilots, 2);

    % 5. Channel Interpolation across all subcarriers
    all_k = (1:simParams.N_fft).';
    p_k   = simParams.pilot_indices(:);
    % Linear 1D interpolation
    h_hat_all = interp1(p_k, avg_h_pilots, all_k, 'linear', 'extrap');

    h_hat_data = h_hat_all(simParams.data_indices);

    % 6. Channel Equalization
    rx_data = rx_freq_grid(simParams.data_indices, :);

    if strcmpi(simParams.equalizer, 'ZF')
        eps_val = 1e-8;
        h_safe = h_hat_data;
        h_safe(abs(h_safe) < eps_val) = eps_val;
        eq_data = rx_data ./ repmat(h_safe, 1, num_ofdm_syms);
    else
        % MMSE Equalizer: W = H* / (|H|^2 + sigma_n^2)
        h_conj = conj(h_hat_data);
        h_pow  = abs(h_hat_data).^2;
        w_mmse = h_conj ./ (h_pow + max(noise_variance, 1e-6));
        eq_data = rx_data .* repmat(w_mmse, 1, num_ofdm_syms);
    end

    eq_symbols = eq_data(:);

    % 7. Constellation Demapping
    norm_factor = simParams.norm_factors(mod_type);
    bps = simParams.bps_map(mod_type);

    switch upper(mod_type)
        case 'BPSK'
            rx_bits_raw = real(eq_symbols) > 0;

        case 'QPSK'
            % De-normalize
            s_scaled = eq_symbols / norm_factor;
            b0 = real(s_scaled) > 0;
            b1 = imag(s_scaled) > 0;
            bit_matrix = [b0, b1].';
            rx_bits_raw = bit_matrix(:);

        case '16-QAM'
            s_scaled = eq_symbols / norm_factor;
            I = real(s_scaled);
            Q = imag(s_scaled);
            N_s = length(eq_symbols);
            rx_bits_raw = zeros(N_s * 4, 1);

            % Invert Gray mapping for 16-QAM:
            % I in [-3, -1, 1, 3] -> decision boundaries at -2, 0, +2
            for n = 1:N_s
                i_val = I(n);
                q_val = Q(n);
                if i_val < -2
                    bi = [0; 0];
                elseif i_val < 0
                    bi = [0; 1];
                elseif i_val < 2
                    bi = [1; 1];
                else
                    bi = [1; 0];
                end

                if q_val < -2
                    bq = [0; 0];
                elseif q_val < 0
                    bq = [0; 1];
                elseif q_val < 2
                    bq = [1; 1];
                else
                    bq = [1; 0];
                end

                idx_start = (n - 1) * 4 + 1;
                rx_bits_raw(idx_start : idx_start + 3) = [bi; bq];
            end

        case '64-QAM'
            s_scaled = eq_symbols / norm_factor;
            I = real(s_scaled);
            Q = imag(s_scaled);
            N_s = length(eq_symbols);
            rx_bits_raw = zeros(N_s * 6, 1);

            % Decision boundaries at -6, -4, -2, 0, 2, 4, 6
            gray_inv = @(val) ...
                (val < -6) * [0;0;0] + ...
                (val >= -6 & val < -4) * [0;0;1] + ...
                (val >= -4 & val < -2) * [0;1;1] + ...
                (val >= -2 & val < 0)  * [0;1;0] + ...
                (val >= 0  & val < 2)  * [1;1;0] + ...
                (val >= 2  & val < 4)  * [1;1;1] + ...
                (val >= 4  & val < 6)  * [1;0;1] + ...
                (val >= 6)             * [1;0;0];

            for n = 1:N_s
                bi = gray_inv(I(n));
                bq = gray_inv(Q(n));
                idx_start = (n - 1) * 6 + 1;
                rx_bits_raw(idx_start : idx_start + 5) = [bi; bq];
            end
    end

    % Remove zero-padding
    if pad_len > 0 && length(rx_bits_raw) >= pad_len
        rx_bits = rx_bits_raw(1 : end - pad_len);
    else
        rx_bits = rx_bits_raw;
    end

    % Align length with transmitted bits
    min_len = min(length(tx_bits), length(rx_bits));
    tx_sub = tx_bits(1:min_len);
    rx_sub = rx_bits(1:min_len);

    % Bit Error Rate
    bit_errors = sum(tx_sub ~= rx_sub);
    ber = bit_errors / min_len;

    % EVM Calculation
    N_data_syms = ceil(min_len / bps);
    eq_sub = eq_symbols(1:min_data_len(eq_symbols, N_data_syms));
    evm_rms = sqrt(mean(abs(eq_sub - mean(eq_sub)).^2)); % baseline proxy

    % 8. Channel Feature Extraction
    h_mag = abs(h_hat_data);
    ch_features.h_mean = mean(h_mag);
    ch_features.h_var  = var(h_mag);
    ch_features.h_min  = min(h_mag);
    ch_features.pilot_error_var = mean(abs(avg_h_pilots - h_hat_all(simParams.pilot_indices)).^2);
end

function n = min_data_len(arr, target)
    n = min(length(arr), target);
end
