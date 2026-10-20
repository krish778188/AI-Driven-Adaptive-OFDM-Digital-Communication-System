function [rx_signal, h_cir, h_cfr, noise_variance] = wireless_channel(tx_signal, snr_dB, channel_type, simParams)
%% WIRELESS_CHANNEL
% Models realistic wireless propagation channels for OFDM:
% 1. Multipath Rayleigh Fading with exponential Power Delay Profile (PDP)
% 2. Additive White Gaussian Noise (AWGN) calibrated to SNR
% 3. Optional Carrier Frequency Offset (CFO)
%
% Inputs:
%   tx_signal    - Time-domain transmitted sample vector
%   snr_dB       - Channel Signal-to-Noise Ratio in dB
%   channel_type - 'AWGN' or 'Rayleigh'
%   simParams    - Simulation parameter structure
%
% Outputs:
%   rx_signal      - Received noisy, faded time-domain signal
%   h_cir          - Discrete Channel Impulse Response h[n]
%   h_cfr          - True Channel Frequency Response H[k] (length N_fft)
%   noise_variance - Noise power sigma_n^2

    tx_signal = tx_signal(:);
    N_fft = simParams.N_fft;

    % 1. Generate Channel Impulse Response (CIR)
    if strcmpi(channel_type, 'AWGN')
        h_cir = 1.0;
        faded_signal = tx_signal;
    else
        % Multipath Rayleigh Fading
        num_taps = simParams.num_taps;
        delays = simParams.tap_delays;
        decay = simParams.decay_factor;

        % Exponential power delay profile
        powers = exp(-decay * (0:num_taps-1));
        powers = powers / sum(powers); % normalize average power to 1

        max_delay = max(delays);
        h_cir = zeros(max_delay + 1, 1);

        for t = 1:num_taps
            d = delays(t) + 1; % 1-based indexing
            % Complex Gaussian tap
            tap_val = sqrt(powers(t)/2) * (randn + 1j * randn);
            h_cir(d) = h_cir(d) + tap_val;
        end

        % Normalize CIR energy of this realization
        h_cir = h_cir / norm(h_cir);

        % Apply channel via linear convolution
        conv_out = conv(tx_signal, h_cir);
        faded_signal = conv_out(1:length(tx_signal));
    end

    % 2. True Channel Frequency Response across subcarriers
    h_cfr = fft(h_cir, N_fft);

    % 3. Carrier Frequency Offset (CFO) if configured
    if isfield(simParams, 'cfo_norm') && abs(simParams.cfo_norm) > 1e-6
        n_idx = (0:length(faded_signal)-1).';
        cfo_rot = exp(1j * 2 * pi * simParams.cfo_norm * n_idx / N_fft);
        faded_signal = faded_signal .* cfo_rot;
    end

    % 4. Additive White Gaussian Noise (AWGN)
    sig_power = mean(abs(faded_signal).^2);
    snr_linear = 10^(snr_dB / 10);
    noise_variance = sig_power / snr_linear;

    % Complex AWGN CN(0, noise_variance)
    noise_std = sqrt(noise_variance / 2);
    noise = noise_std * (randn(size(faded_signal)) + 1j * randn(size(faded_signal)));

    rx_signal = faded_signal + noise;
end
