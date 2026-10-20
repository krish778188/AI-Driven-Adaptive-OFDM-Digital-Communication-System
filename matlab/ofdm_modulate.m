function [tx_signal, tx_symbols, num_ofdm_syms, pad_len] = ofdm_modulate(bits, mod_type, simParams)
%% OFDM_MODULATE
% Implements the transmitter chain of the OFDM communication system:
% 1. Binary serial-to-parallel conversion & zero-padding
% 2. Gray-coded constellation mapping (BPSK, QPSK, 16-QAM, 64-QAM)
% 3. Unit-energy power normalization
% 4. Subcarrier grid assembly with deterministic pilots and DC/guard nulls
% 5. IFFT modulation with energy-preserving normalization
% 6. Cyclic Prefix (CP) insertion and serialization
%
% Inputs:
%   bits       - Binary vector of input bits (0s and 1s)
%   mod_type   - 'BPSK', 'QPSK', '16-QAM', or '64-QAM'
%   simParams  - System parameters structure from initialize_simulation.m
%
% Outputs:
%   tx_signal      - Serialized time-domain baseband samples with CP
%   tx_symbols     - Complex modulated data symbols
%   num_ofdm_syms  - Number of OFDM symbols synthesized
%   pad_len        - Padding bits added to achieve integer symbol alignment

    bits = bits(:);
    bps = simParams.bps_map(mod_type);
    norm_factor = simParams.norm_factors(mod_type);

    % Zero-pad bits if not a multiple of bits per symbol (bps)
    rem_bits = mod(length(bits), bps);
    pad_len = 0;
    if rem_bits ~= 0
        pad_len = bps - rem_bits;
        bits = [bits; zeros(pad_len, 1)];
    end

    % 1. Digital Constellation Mapping
    switch upper(mod_type)
        case 'BPSK'
            % '0' -> -1, '1' -> +1
            tx_symbols = 2 * bits - 1;

        case 'QPSK'
            % Gray mapping: 2 bits [b0, b1] -> (I + j*Q)
            bit_pairs = reshape(bits, 2, []).';
            I = 2 * bit_pairs(:, 1) - 1;
            Q = 2 * bit_pairs(:, 2) - 1;
            tx_symbols = (I + 1j * Q) * norm_factor;

        case '16-QAM'
            % Gray mapping: 4 bits [b0 b1 b2 b3] -> I in {-3,-1,1,3}, Q in {-3,-1,1,3}
            bit_quads = reshape(bits, 4, []).';
            % Gray mapping table:
            % 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
            gray_map = containers.Map({'00', '01', '11', '10'}, {-3, -1, 1, 3});
            N_s = size(bit_quads, 1);
            I = zeros(N_s, 1);
            Q = zeros(N_s, 1);
            for n = 1:N_s
                i_str = sprintf('%d%d', bit_quads(n, 1), bit_quads(n, 2));
                q_str = sprintf('%d%d', bit_quads(n, 3), bit_quads(n, 4));
                I(n) = gray_map(i_str);
                Q(n) = gray_map(q_str);
            end
            tx_symbols = (I + 1j * Q) * norm_factor;

        case '64-QAM'
            bit_six = reshape(bits, 6, []).';
            gray_map64 = containers.Map(...
                {'000', '001', '011', '010', '110', '111', '101', '100'}, ...
                {-7, -5, -3, -1, 1, 3, 5, 7});
            N_s = size(bit_six, 1);
            I = zeros(N_s, 1);
            Q = zeros(N_s, 1);
            for n = 1:N_s
                i_str = sprintf('%d%d%d', bit_six(n, 1), bit_six(n, 2), bit_six(n, 3));
                q_str = sprintf('%d%d%d', bit_six(n, 4), bit_six(n, 5), bit_six(n, 6));
                I(n) = gray_map64(i_str);
                Q(n) = gray_map64(q_str);
            end
            tx_symbols = (I + 1j * Q) * norm_factor;

        otherwise
            error('Unsupported modulation type: %s', mod_type);
    end

    % 2. Subcarrier Grid Allocation
    N_data = simParams.N_data;
    num_data_syms = length(tx_symbols);
    num_ofdm_syms = ceil(num_data_syms / N_data);

    % Zero pad symbols to fit full OFDM symbol blocks
    pad_syms = (num_ofdm_syms * N_data) - num_data_syms;
    if pad_syms > 0
        tx_symbols_padded = [tx_symbols; zeros(pad_syms, 1)];
    else
        tx_symbols_padded = tx_symbols;
    end

    % Reshape into (N_data x num_ofdm_syms)
    data_grid = reshape(tx_symbols_padded, N_data, num_ofdm_syms);

    % Full subcarrier matrix (N_fft x num_ofdm_syms)
    freq_grid = zeros(simParams.N_fft, num_ofdm_syms);
    freq_grid(simParams.data_indices, :)  = data_grid;
    freq_grid(simParams.pilot_indices, :) = repmat(simParams.pilot_values, 1, num_ofdm_syms);

    % 3. Normalized IFFT (Unitary energy preservation)
    time_blocks = ifft(freq_grid, simParams.N_fft, 1) * sqrt(simParams.N_fft);

    % 4. Cyclic Prefix Insertion
    cp = time_blocks(end - simParams.N_cp + 1 : end, :);
    time_blocks_cp = [cp; time_blocks];

    % 5. Serialization into 1D continuous baseband stream
    tx_signal = time_blocks_cp(:);
end
