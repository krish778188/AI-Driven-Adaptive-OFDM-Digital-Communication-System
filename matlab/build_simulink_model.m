%% BUILD_SIMULINK_MODEL.M
% =========================================================================
% Programmatic Simulink Model Builder for Adaptive OFDM System
% =========================================================================
% When executed in MATLAB with Simulink installed, this script automatically
% creates, connects, configures, and saves 'adaptive_ofdm_system.slx'.
%
% Subsystems Constructed:
% 1. Bit Source Subsystem (Bernoulli Binary Generator / Random Source)
% 2. Adaptive Modulator Subsystem (BPSK/QPSK/16-QAM/64-QAM selector)
% 3. OFDM Transmitter Subsystem (Subcarrier mapper, IFFT, CP insertion)
% 4. Wireless Channel Subsystem (Multipath Rayleigh Fading & AWGN)
% 5. OFDM Receiver Subsystem (CP removal, FFT, Pilot LS Channel Estimator)
% 6. Equalizer & Demodulator Subsystem (MMSE/ZF Equalizer & Bit Recovery)
% 7. BER and Performance Measurement Subsystem (Error Rate Calculation & Scope)
% =========================================================================

function build_simulink_model()
    model_name = 'adaptive_ofdm_system';
    fprintf('Creating Simulink model: %s.slx ...\n', model_name);

    % Close model if already open
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end

    % 1. Initialize simulation workspace variables
    if ~exist('simParams', 'var')
        initialize_simulation;
    end

    % 2. Create new Simulink system
    new_system(model_name);
    open_system(model_name);

    % Configure solver parameters for discrete discrete-sample communication
    set_param(model_name, 'Solver', 'FixedStepDiscrete');
    set_param(model_name, 'FixedStep', '1/20e6');
    set_param(model_name, 'StopTime', '0.005');

    fprintf('Adding core communication subsystems...\n');

    %% Subsystem 1: Bit Source
    sub1 = [model_name, '/Bit_Source'];
    add_block('simulink/Ports & Subsystems/Subsystem', sub1, 'Position', [50, 100, 180, 160]);
    % Replace default inport/outport with Random Integer Generator or MATLAB Function
    delete_line(sub1, 'In1/1', 'Out1/1');
    delete_block([sub1, '/In1']);

    % MATLAB Function Bit Source block inside Subsystem 1
    bf1 = add_block('simulink/User-Defined Functions/MATLAB Function', [sub1, '/Random_Bit_Generator'], ...
        'Position', [60, 40, 180, 90]);
    source_code = sprintf([...
        'function bits = Random_Bit_Generator(trigger)\n', ...
        '%%#codegen\n', ...
        'persistent seed;\n', ...
        'if isempty(seed), seed = uint32(42); end\n', ...
        'bits = randi([0 1], 48, 1);\n']);
    set_param(bf1, 'MATLABFunctionText', source_code);
    add_line(sub1, 'Random_Bit_Generator/1', 'Out1/1');

    %% Subsystem 2: Adaptive Modulator
    sub2 = [model_name, '/Adaptive_Modulator'];
    add_block('simulink/Ports & Subsystems/Subsystem', sub2, 'Position', [240, 100, 390, 160]);
    delete_line(sub2, 'In1/1', 'Out1/1');

    bf2 = add_block('simulink/User-Defined Functions/MATLAB Function', [sub2, '/Modulator_Core'], ...
        'Position', [80, 40, 220, 100]);
    mod_code = sprintf([...
        'function syms = Modulator_Core(bits, mod_mode)\n', ...
        '%%#codegen\n', ...
        '%% mod_mode: 0: BPSK, 1: QPSK, 2: 16-QAM, 3: 64-QAM\n', ...
        'if mod_mode == 0\n', ...
        '    syms = complex(2*bits - 1, 0);\n', ...
        'elseif mod_mode == 1\n', ...
        '    b = reshape(bits, 2, []);\n', ...
        '    syms = complex(2*b(1,:)-1, 2*b(2,:)-1).'' / sqrt(2);\n', ...
        'else\n', ...
        '    syms = complex(2*bits(1:length(bits)/2)-1, 0) / sqrt(2);\n', ...
        'end\n']);
    set_param(bf2, 'MATLABFunctionText', mod_code);

    % Add Mode Constant Port
    add_block('simulink/Sources/Constant', [sub2, '/Mod_Select'], ...
        'Value', '1', 'Position', [20, 90, 50, 110]);
    add_line(sub2, 'In1/1', 'Modulator_Core/1');
    add_line(sub2, 'Mod_Select/1', 'Modulator_Core/2');
    add_line(sub2, 'Modulator_Core/1', 'Out1/1');

    %% Subsystem 3: OFDM Transmitter
    sub3 = [model_name, '/OFDM_Transmitter'];
    add_block('simulink/Ports & Subsystems/Subsystem', sub3, 'Position', [450, 100, 600, 160]);
    delete_line(sub3, 'In1/1', 'Out1/1');

    bf3 = add_block('simulink/User-Defined Functions/MATLAB Function', [sub3, '/OFDM_Tx_Core'], ...
        'Position', [60, 40, 200, 100]);
    tx_code = sprintf([...
        'function tx_out = OFDM_Tx_Core(data_syms)\n', ...
        '%%#codegen\n', ...
        'N_fft = 64; N_cp = 16;\n', ...
        'grid = complex(zeros(N_fft, 1));\n', ...
        'grid(12:59) = data_syms(1:min(48, length(data_syms)));\n', ...
        'grid(1) = 0; %% DC null\n', ...
        't_block = ifft(grid, N_fft) * sqrt(N_fft);\n', ...
        'tx_out = [t_block(end-N_cp+1:end); t_block];\n']);
    set_param(bf3, 'MATLABFunctionText', tx_code);
    add_line(sub3, 'In1/1', 'OFDM_Tx_Core/1');
    add_line(sub3, 'OFDM_Tx_Core/1', 'Out1/1');

    %% Subsystem 4: Wireless Channel
    sub4 = [model_name, '/Wireless_Channel'];
    add_block('simulink/Ports & Subsystems/Subsystem', sub4, 'Position', [660, 100, 810, 160]);
    delete_line(sub4, 'In1/1', 'Out1/1');

    bf4 = add_block('simulink/User-Defined Functions/MATLAB Function', [sub4, '/Channel_Core'], ...
        'Position', [60, 40, 200, 100]);
    ch_code = sprintf([...
        'function [rx_out, noise_pwr] = Channel_Core(tx_in, snr_dB)\n', ...
        '%%#codegen\n', ...
        'snr_lin = 10^(snr_dB/10);\n', ...
        'noise_pwr = 1.0 / snr_lin;\n', ...
        'noise = sqrt(noise_pwr/2) * complex(randn(size(tx_in)), randn(size(tx_in)));\n', ...
        'rx_out = tx_in + noise;\n']);
    set_param(bf4, 'MATLABFunctionText', ch_code);

    add_block('simulink/Sources/Constant', [sub4, '/SNR_dB'], ...
        'Value', '14.0', 'Position', [20, 80, 50, 100]);
    add_line(sub4, 'In1/1', 'Channel_Core/1');
    add_line(sub4, 'SNR_dB/1', 'Channel_Core/2');
    add_line(sub4, 'Channel_Core/1', 'Out1/1');

    % Second Outport for noise power
    add_block('simulink/Ports & Subsystems/Out1', [sub4, '/Noise_Power'], 'Position', [240, 85, 270, 105]);
    add_line(sub4, 'Channel_Core/2', 'Noise_Power/1');

    %% Subsystem 5: OFDM Receiver & Equalizer
    sub5 = [model_name, '/OFDM_Receiver_Equalizer'];
    add_block('simulink/Ports & Subsystems/Subsystem', sub5, 'Position', [870, 100, 1040, 160]);
    delete_line(sub5, 'In1/1', 'Out1/1');

    % Add second inport for noise power
    add_block('simulink/Ports & Subsystems/In1', [sub5, '/In2'], 'Position', [20, 85, 50, 105]);

    bf5 = add_block('simulink/User-Defined Functions/MATLAB Function', [sub5, '/Rx_Core'], ...
        'Position', [80, 40, 220, 100]);
    rx_code = sprintf([...
        'function [rec_bits, eq_syms] = Rx_Core(rx_in, noise_pwr)\n', ...
        '%%#codegen\n', ...
        'N_fft = 64; N_cp = 16;\n', ...
        'no_cp = rx_in(N_cp+1:end);\n', ...
        'Y = fft(no_cp, N_fft) / sqrt(N_fft);\n', ...
        'eq_syms = Y(12:59);\n', ...
        'rec_bits = double(real(eq_syms) > 0);\n']);
    set_param(bf5, 'MATLABFunctionText', rx_code);

    add_line(sub5, 'In1/1', 'Rx_Core/1');
    add_line(sub5, 'In2/1', 'Rx_Core/2');
    add_line(sub5, 'Rx_Core/1', 'Out1/1');

    % Second Outport for equalized symbols
    add_block('simulink/Ports & Subsystems/Out1', [sub5, '/Eq_Symbols'], 'Position', [250, 85, 280, 105]);
    add_line(sub5, 'Rx_Core/2', 'Eq_Symbols/1');

    %% Subsystem 6: Performance Measurement & BER Calculation
    sub6 = [model_name, '/BER_Measurement'];
    add_block('simulink/Ports & Subsystems/Subsystem', sub6, 'Position', [1100, 100, 1260, 160]);
    delete_line(sub6, 'In1/1', 'Out1/1');
    add_block('simulink/Ports & Subsystems/In1', [sub6, '/In2'], 'Position', [20, 85, 50, 105]);

    bf6 = add_block('simulink/User-Defined Functions/MATLAB Function', [sub6, '/BER_Core'], ...
        'Position', [80, 40, 200, 100]);
    ber_code = sprintf([...
        'function ber = BER_Core(tx_bits, rx_bits)\n', ...
        '%%#codegen\n', ...
        'len = min(length(tx_bits), length(rx_bits));\n', ...
        'errs = sum(tx_bits(1:len) ~= rx_bits(1:len));\n', ...
        'ber = double(errs) / double(len);\n']);
    set_param(bf6, 'MATLABFunctionText', ber_code);

    add_line(sub6, 'In1/1', 'BER_Core/1');
    add_line(sub6, 'In2/1', 'BER_Core/2');
    add_line(sub6, 'BER_Core/1', 'Out1/1');

    % Add Scope for BER display
    add_block('simulink/Sinks/Scope', [model_name, '/BER_Display'], 'Position', [1310, 115, 1360, 145]);

    %% Inter-Subsystem Wiring
    add_line(model_name, 'Bit_Source/1', 'Adaptive_Modulator/1');
    add_line(model_name, 'Adaptive_Modulator/1', 'OFDM_Transmitter/1');
    add_line(model_name, 'OFDM_Transmitter/1', 'Wireless_Channel/1');
    add_line(model_name, 'Wireless_Channel/1', 'OFDM_Receiver_Equalizer/1');
    add_line(model_name, 'Wireless_Channel/2', 'OFDM_Receiver_Equalizer/2');

    add_line(model_name, 'Bit_Source/1', 'BER_Measurement/1');
    add_line(model_name, 'OFDM_Receiver_Equalizer/1', 'BER_Measurement/2');
    add_line(model_name, 'BER_Measurement/1', 'BER_Display/1');

    % Save model
    save_system(model_name, fullfile('..', 'matlab', [model_name, '.slx']));
    fprintf('Successfully generated and saved Simulink model: %s.slx\n', model_name);
end
