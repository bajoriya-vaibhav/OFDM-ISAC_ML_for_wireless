clc; clear; close all;

%% System Parameters (reference from table given in paper)
d0            = 40;             % UE-BS distance [m]
theta_true    = 45*pi/180;      % True AoA [rad]
fc            = 1.8e9;          % Center frequency [Hz]
c_light       = 3e8;            % Speed of light [m/s]
lambda        = c_light/fc;     % Wavelength [m]
NFFT          = 2048;           % FFT size
NCP           = 256;            % CP length
J             = 16;             % BS antennas (ULA)
r_ant         = lambda/2;       % Antenna spacing
alpha_pl      = 2;              % Path loss exponent
L             = 10;             % Total multipath taps
dtheta_max    = 10*pi/180;      % Max angle deviation [rad]

% 20% of taps are external interference
L_interf      = round(0.2*L);   % = 2 interference taps
L_local       = L - L_interf;   % = 8 locally scattered taps

% Index modulation: 6 bits
q_bits        = 6;
Q             = 2^q_bits;       % 64 candidate sequences

% ZC: Ns=839, NCS=13  |  m-seq/Gold: Ns=1023, NCS=15
Ns_ZC         = 839;   NCS_ZC  = 13;
Ns_mG         = 1023;  NCS_mG  = 15;

% SNR range
SNR_dB_vec    = 13:1:20;
num_snr       = length(SNR_dB_vec);

% Monte Carlo trials
num_trials_BER  = 10000;  % Adjust to 30000 or 50000 for smoother curves

fprintf('OFDM-ISAC BER Communication Simulation Starting...\n');
fprintf('Parameters: d0=%dm, J=%d antennas, L=%d paths\n', d0, J, L);

%% Gray code table (Gray mapping for 64-PSK)
gray_enc = zeros(1,Q);
for i=0:Q-1
    gray_enc(i+1) = bitxor(i, floor(i/2));
end
idx_to_bits = zeros(Q, q_bits);
for i=0:Q-1
    g = gray_enc(i+1);
    idx_to_bits(i+1,:) = de2bi(g, q_bits, 'left-msb');
end

%% Orthogonal sequence generation

% Zadoff-Chu sequence (root u=1)
u = 1;
n_zc = (0:Ns_ZC-1)';
s0_ZC = exp(-1j*pi*u*n_zc.*(n_zc+1)/Ns_ZC);

% m-sequence (length 1023 = 2^10-1, poly x^10+x^7+1)
s0_mseq = generate_mseq(10, [10,7]);

% Gold sequence (XOR of two preferred-pair m-sequences)
m1 = generate_mseq(10, [10,7]);
m2 = generate_mseq(10, [10,3]);      % Preferred pair: x^10+x^3+1
gold_bin = mod(((m1+1)/2) + ((m2+1)/2), 2);
s0_gold  = 2*gold_bin - 1;

% Generate Q candidate sequences by cyclic shift (Eq. 1)
S_ZC   = make_candidates(s0_ZC,   NCS_ZC,  Q);
S_mseq = make_candidates(s0_mseq, NCS_mG,  Q);
S_gold = make_candidates(s0_gold, NCS_mG,  Q);

%% BER simulation (30.72 MHz bandwidth)
fprintf('\n--- Fig.4: BER Simulation (30.72 MHz bandwidth) ---\n');
BW_ber = 30.72e6;
Ts_ber = 1/BW_ber;

BER_ZC   = zeros(1,num_snr);
BER_mseq = zeros(1,num_snr);
BER_gold = zeros(1,num_snr);

for si = 1:num_snr
    SNR_dB  = SNR_dB_vec(si);
    SNR_lin = 10^(SNR_dB/10);
    err_ZC=0; err_mseq=0; err_gold=0;

    for trial = 1:num_trials_BER
        i_tx = randi([0,Q-1]);
        tx_bits = idx_to_bits(i_tx+1,:);

        err_ZC = err_ZC + run_ber_trial(S_ZC, s0_ZC, Ns_ZC, NCS_ZC,...
            NFFT, NCP, J, d0, theta_true, lambda, r_ant, alpha_pl,...
            L_local, L_interf, dtheta_max, Ts_ber, c_light, SNR_lin,...
            i_tx, tx_bits, idx_to_bits, Q, q_bits);

        err_mseq = err_mseq + run_ber_trial(S_mseq, s0_mseq, Ns_mG, NCS_mG,...
            NFFT, NCP, J, d0, theta_true, lambda, r_ant, alpha_pl,...
            L_local, L_interf, dtheta_max, Ts_ber, c_light, SNR_lin,...
            i_tx, tx_bits, idx_to_bits, Q, q_bits);

        err_gold = err_gold + run_ber_trial(S_gold, s0_gold, Ns_mG, NCS_mG,...
            NFFT, NCP, J, d0, theta_true, lambda, r_ant, alpha_pl,...
            L_local, L_interf, dtheta_max, Ts_ber, c_light, SNR_lin,...
            i_tx, tx_bits, idx_to_bits, Q, q_bits);
    end

    BER_ZC(si)   = err_ZC   / (num_trials_BER * q_bits);
    BER_mseq(si) = err_mseq / (num_trials_BER * q_bits);
    BER_gold(si) = err_gold / (num_trials_BER * q_bits);

    fprintf('SNR=%2ddB | ZC=%.2e | mseq=%.2e | Gold=%.2e\n',...
        SNR_dB, BER_ZC(si), BER_mseq(si), BER_gold(si));
end

%% Plotting Fig. 4 - BER
figure('Position',[100,100,650,500],'Color','w');
semilogy(SNR_dB_vec, BER_ZC,   'k-s',  'LineWidth',1.8,'MarkerSize',7); hold on;
semilogy(SNR_dB_vec, BER_gold, 'b-o',  'LineWidth',1.8,'MarkerSize',7);
semilogy(SNR_dB_vec, BER_mseq, 'r-^',  'LineWidth',1.8,'MarkerSize',7);
grid on;
grid minor;
set(gca, 'GridColor', [0.15 0.15 0.15], 'GridAlpha', 0.5);
set(gca, 'MinorGridColor', [0.15 0.15 0.15], 'MinorGridAlpha', 0.2);
xlabel('Transmit SNR [dB]','FontSize',13);
ylabel('BER','FontSize',13);
legend('Zadoff-Chu sequence','gold sequence','m-sequence',...
       'Location','southwest','FontSize',11);
xlim([13,20]); ylim([1e-5,1e0]);
set(gca,'FontSize',12,'YTick',[1e-5,1e-4,1e-3,1e-2,1e-1,1e0], 'Color', 'w');
set(gcf, 'Color', 'w');
title('Fig. 4: BER Performance','FontSize',13);

fprintf('\nBER Simulation complete.\n');

function mseq = generate_mseq(n, poly_exponents)
    N   = 2^n - 1;
    reg = ones(1,n);
    mseq = zeros(1,N);
    exponents = poly_exponents(poly_exponents < n);
    tap_positions = n - exponents;
    tap_positions = [tap_positions, n];
    for k=1:N
        mseq(k) = reg(end);
        fb = mod(sum(reg(tap_positions)), 2);
        reg = [fb, reg(1:end-1)];
    end
    mseq = 2*mseq - 1;
end

function S = make_candidates(s0, NCS, Q)
    Ns = length(s0);
    S  = zeros(Q, Ns);
    for i=0:Q-1
        S(i+1,:) = circshift(s0(:).', i*NCS);
    end
end

function [h_delay, tau_int] = gen_channel(d0, theta, lambda, r_ant, J,...
    L_local, L_interf, dtheta_max, alpha_pl, Ts, c_light, ~)
    L = L_local + L_interf;
    h_delay  = zeros(J, L);
    tau_int  = zeros(1, L);
    tau0 = round(d0 / (c_light * Ts));
    for l = 1:L
        tau_int(l) = tau0 + (l - 1);
        dl = d0;
        if l <= L_local
            dtheta  = (2*rand()-1) * dtheta_max;
            theta_l = theta + dtheta;
        else
            theta_l = (rand()-0.5) * pi;
        end
        beta_l = (randn + 1j*randn) / sqrt(2);
        for jj = 1:J
            phase = -1j*2*pi/lambda * (jj-1) * r_ant * cos(theta_l);
            h_delay(jj, l) = beta_l * exp(phase) * sqrt(dl^(-alpha_pl));
        end
    end
end

function xCP = ofdm_tx(s_i, Ns, NFFT, NCP)
    S_freq     = fft(s_i(:).', Ns);
    S_map      = zeros(1, NFFT);
    S_map(1:Ns) = S_freq;
    x          = ifft(S_map, NFFT);
    xCP        = [x(NFFT-NCP+1:NFFT), x];
end

function y_all = ofdm_rx(r_sig, J, Ns, NFFT, NCP)
    y_all = zeros(J, Ns);
    for jj=1:J
        r_nocp = r_sig(jj, NCP+1 : NCP+NFFT);
        R_freq = fft(r_nocp, NFFT);
        R_sub  = R_freq(1:Ns);
        y_all(jj,:) = ifft(R_sub, Ns);
    end
end

function r_sig = apply_channel(xCP, h_delay, tau_int, J, L, NFFT, NCP, noise_var)
    r_sig = zeros(J, NFFT+NCP);
    sig_len = NFFT+NCP;
    for jj=1:J
        r_j = zeros(1, sig_len);
        for l=1:L
            tau_l = tau_int(l);
            if tau_l < sig_len
                xCP_del = [zeros(1,tau_l), xCP(1:sig_len-tau_l)];
                r_j = r_j + h_delay(jj,l) * xCP_del;
            end
        end
        noise = sqrt(noise_var/2)*(randn(1,sig_len)+1j*randn(1,sig_len));
        r_sig(jj,:) = r_j + noise;
    end
end

function [i_hat, kappa_hat_all] = detect_seq(y_all, s0, NCS, J, Q, use_diversity)
    if nargin < 6, use_diversity = true; end
    Ns = length(s0);
    i_per_ant     = zeros(1,J);
    kappa_per_ant = zeros(1,J);
    S0_fft = fft(s0(:).', Ns);
    for jj=1:J
        Yj   = fft(y_all(jj,:), Ns);
        R_j  = abs(ifft(Yj .* conj(S0_fft)));
        [~, idx]  = max(R_j);
        kappa_hat = idx - 1;
        i_j = floor(kappa_hat / NCS);
        i_j = min(i_j, Q-1);
        i_per_ant(jj)     = i_j;
        kappa_per_ant(jj) = kappa_hat;
    end
    if use_diversity && J > 1
        i_hat = mode(i_per_ant);
    else
        i_hat = i_per_ant(1);
    end
    kappa_hat_all = kappa_per_ant;
end

function num_errors = run_ber_trial(S_cand, s0, Ns, NCS,...
    NFFT, NCP, J, d0, theta, lambda, r_ant, alpha_pl,...
    L_local, L_interf, dtheta_max, Ts, c_light, SNR_lin,...
    i_tx, tx_bits, idx_to_bits, Q, q_bits)

    s_i = S_cand(i_tx+1,:);
    xCP = ofdm_tx(s_i, Ns, NFFT, NCP);

    noise_var = 5.0 * (NFFT + NCP) / (Ns * SNR_lin);

    [h_delay, tau_int] = gen_channel(d0, theta, lambda, r_ant, J,...
        L_local, L_interf, dtheta_max, alpha_pl, Ts, c_light, NCP);

    r_sig = apply_channel(xCP, h_delay, tau_int, J, L_local+L_interf,...
        NFFT, NCP, noise_var);

    y_all = ofdm_rx(r_sig, J, Ns, NFFT, NCP);

    [i_hat, ~] = detect_seq(y_all, s0, NCS, J, Q, true);
    i_hat = max(0, min(i_hat, Q-1));

    rx_bits = idx_to_bits(i_hat+1,:);
    num_errors = sum(rx_bits ~= tx_bits);
end
