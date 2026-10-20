%% PLOT_RESULTS.M
% =========================================================================
% Publication-Quality MATLAB Plotting Script for Adaptive OFDM System
% =========================================================================

results_dir = fullfile('..', 'results');
mat_file = fullfile(results_dir, 'matlab_simulation_results.mat');

if ~exist(mat_file, 'file')
    fprintf('Results file not found. Running simulation first...\n');
    run_simulation;
    return;
end

load(mat_file);

colors = {
    [0.12, 0.47, 0.71], ... % BPSK: Blue
    [0.17, 0.63, 0.17], ... % QPSK: Green
    [1.00, 0.50, 0.05], ... % 16-QAM: Orange
    [0.84, 0.15, 0.16], ... % 64-QAM: Red
    [0.58, 0.40, 0.74]      % Adaptive: Purple
};

markers = {'o', 's', '^', 'd', 'p'};

%% Figure 1: BER vs SNR
fig1 = figure('Name', 'BER vs SNR Curves', 'Color', 'w', 'Position', [100, 100, 750, 500]);
hold on; grid on; box on;

for m = 1:length(schemes)
    line_w = 1.8;
    if strcmp(schemes{m}, 'Adaptive')
        line_w = 2.8;
    end
    % Clamp zero BER to 1e-6 for log plot
    ber_p = ber_matrix(m, :);
    ber_p(ber_p == 0) = 1e-6;

    semilogy(snr_range, ber_p, ...
        'Color', colors{m}, ...
        'Marker', markers{m}, ...
        'LineWidth', line_w, ...
        'MarkerSize', 7, ...
        'MarkerFaceColor', colors{m}, ...
        'DisplayName', schemes{m});
end

% Target BER line
yline(simParams.target_ber, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Target BER (%.1e)', simParams.target_ber));

set(gca, 'YScale', 'log');
ylim([1e-5, 1]);
xlim([min(snr_range), max(snr_range)]);
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('OFDM Bit Error Rate vs. SNR (%s Channel)', simParams.default_channel), 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 10);

saveas(fig1, fullfile(results_dir, 'matlab_ber_vs_snr.png'));

%% Figure 2: Net Effective Throughput vs SNR
fig2 = figure('Name', 'Throughput vs SNR', 'Color', 'w', 'Position', [150, 150, 750, 500]);
hold on; grid on; box on;

for m = 1:length(schemes)
    line_w = 1.8;
    if strcmp(schemes{m}, 'Adaptive')
        line_w = 3.0;
    end
    plot(snr_range, throughput_matrix(m, :), ...
        'Color', colors{m}, ...
        'Marker', markers{m}, ...
        'LineWidth', line_w, ...
        'MarkerSize', 7, ...
        'MarkerFaceColor', colors{m}, ...
        'DisplayName', schemes{m});
end

xlim([min(snr_range), max(snr_range)]);
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Net Effective Throughput (Mbps)', 'FontSize', 12, 'FontWeight', 'bold');
title('Net Throughput vs. SNR (20 MHz Channel Bandwidth)', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);

saveas(fig2, fullfile(results_dir, 'matlab_throughput_vs_snr.png'));

fprintf('Plots successfully saved to: %s\n', results_dir);
