%% =========================================================================
%  OFDM-ISAC Complete Simulation
%
%  Key equations implemented:
%  Eq.(1): Candidate sequence generation via cyclic shift
%  Eq.(3)-(5): OFDM TX (DFT -> subcarrier map -> IFFT -> CP)
%  Eq.(6): Tapped delay line channel model
%  Eq.(8)-(9): OFDM RX (CP removal -> FFT -> demap -> IDFT)
%  Eq.(11): Correlation profile
%  Eq.(12): Sequence detection
%  Eq.(14): Distance estimation
%  Eq.(15)-(16): Direction estimation (despreading + spatial correlation)
%  Algorithm 1: DBSCAN data collection
%  Eq.(17): Clustering-based localization
%  Eq.(18): RMSE definition
%% =========================================================================
clc; clear; close all;

%% =========================================================================
%  SYSTEM PARAMETERS (Table I)
%% =========================================================================
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
eta           = 1;              % Power threshold for DBSCAN

% 20% of taps are external interference
L_interf      = round(0.2*L);   % = 2 interference taps
L_local       = L - L_interf;   % = 8 locally scattered taps

% Index modulation: 6 bits
q_bits        = 6;
Q             = 2^q_bits;       % 64 candidate sequences

% ZC: Ns=839, NCS=13  |  m-seq/Gold: Ns=1023, NCS=15
Ns_ZC         = 839;   NCS_ZC  = 13;
Ns_mG         = 1023;  NCS_mG  = 15;

% Angle search grid (P angles)
P             = 361;
angle_grid    = linspace(-pi/2, pi/2, P);  % [-90, 90] degrees

% SNR range
SNR_dB_vec    = 13:1:20;
num_snr       = length(SNR_dB_vec);

% Monte Carlo trials
num_trials_BER  = 10000;  
num_trials_RMSE = 1000;

% DBSCAN uses 5*NCS data samples (5 sequences)
num_seq_dbscan  = 5;

fprintf('OFDM-ISAC Simulation Starting...\n');
fprintf('Parameters: d0=%dm, theta=%ddeg, J=%d antennas, L=%d paths\n',...
    d0, round(theta_true*180/pi), J, L);

%% =========================================================================
%  GRAY CODE TABLE (Gray mapping for 64-PSK)
%% =========================================================================
gray_enc = zeros(1,Q);
for i=0:Q-1
    gray_enc(i+1) = bitxor(i, floor(i/2));
end
gray_dec = zeros(1,Q);
for i=0:Q-1
    gray_dec(gray_enc(i+1)+1) = i;
end
idx_to_bits = zeros(Q, q_bits);
for i=0:Q-1
    g = gray_enc(i+1);
    idx_to_bits(i+1,:) = de2bi(g, q_bits, 'left-msb');
end

%% =========================================================================
%  ORTHOGONAL SEQUENCE GENERATION
%% =========================================================================

% --- Zadoff-Chu sequence (root u=1) ---
u = 1;
n_zc = (0:Ns_ZC-1)';
s0_ZC = exp(-1j*pi*u*n_zc.*(n_zc+1)/Ns_ZC);

% --- m-sequence (length 1023 = 2^10-1, poly x^10+x^7+1) ---
s0_mseq = generate_mseq(10, [10,7]);

% --- Gold sequence (XOR of two preferred-pair m-sequences) ---
m1 = generate_mseq(10, [10,7]);
m2 = generate_mseq(10, [10,3]);      % Preferred pair: x^10+x^3+1
gold_bin = mod(((m1+1)/2) + ((m2+1)/2), 2);
s0_gold  = 2*gold_bin - 1;

% --- Generate Q candidate sequences by cyclic shift (Eq. 1) ---
S_ZC   = make_candidates(s0_ZC,   NCS_ZC,  Q);
S_mseq = make_candidates(s0_mseq, NCS_mG,  Q);
S_gold = make_candidates(s0_gold, NCS_mG,  Q);

fprintf('Sequences generated: ZC(%d), m-seq(%d), Gold(%d)\n',...
    Ns_ZC, Ns_mG, Ns_mG);

%% =========================================================================
%  STEERING VECTOR MATRIX A (J x P) for direction finding (Eq. 16)
%% =========================================================================
A = zeros(J,P);
for p=1:P
    for jj=1:J
        A(jj,p) = exp(-1j*2*pi/lambda*(jj-1)*r_ant*cos(angle_grid(p)));
    end
end

%% =========================================================================
%  FIG. 4: BER SIMULATION — BW=30.72MHz, delta_f=15kHz
%% =========================================================================
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

% %% =========================================================================
% %  FIG. 5: RMSE SIMULATION (ZC sequence, three bandwidths)
% %  [COMMENTED OUT — will fix after BER is finalized]
% %% =========================================================================
% fprintf('\n--- Fig.5: RMSE Simulation (ZC sequence, 3 bandwidths) ---\n');
% BW_vec    = [30.72e6, 61.44e6, 184.32e6];
% num_BW    = length(BW_vec);
% RMSE_all  = zeros(num_BW, num_snr);
% 
% for bw_idx = 1:num_BW
%     BW_cur = BW_vec(bw_idx);
%     Ts_cur = 1/BW_cur;
%     fprintf('\nBandwidth = %.2f MHz\n', BW_cur/1e6);
% 
%     for si = 1:num_snr
%         SNR_dB  = SNR_dB_vec(si);
%         SNR_lin = 10^(SNR_dB/10);
%         rmse_sq_sum = 0;
%         valid_count = 0;
% 
%         for trial = 1:num_trials_RMSE
%             i_tx = randi([0,Q-1]);
% 
%             [d0_hat, theta_hat, success] = run_rmse_trial(...
%                 S_ZC, s0_ZC, Ns_ZC, NCS_ZC,...
%                 NFFT, NCP, J, d0, theta_true, lambda, r_ant, alpha_pl,...
%                 L_local, L_interf, dtheta_max, Ts_cur, c_light, SNR_lin,...
%                 i_tx, Q, A, angle_grid, num_seq_dbscan, eta);
% 
%             if success
%                 err_x = d0_hat*cos(theta_hat) - d0*cos(theta_true);
%                 err_y = d0_hat*sin(theta_hat) - d0*sin(theta_true);
%                 rmse_sq_sum = rmse_sq_sum + err_x^2 + err_y^2;
%                 valid_count = valid_count + 1;
%             end
%         end
% 
%         if valid_count > 0
%             RMSE_all(bw_idx,si) = sqrt(rmse_sq_sum / valid_count);
%         else
%             RMSE_all(bw_idx,si) = NaN;
%         end
% 
%         fprintf('  SNR=%2ddB | RMSE=%.2f m (valid=%d/%d)\n',...
%             SNR_dB, RMSE_all(bw_idx,si), valid_count, num_trials_RMSE);
%     end
% end

%% =========================================================================
%  PLOT FIG. 4 - BER
%% =========================================================================
figure('Position',[100,100,650,500],'Color','w');
semilogy(SNR_dB_vec, BER_ZC,   'k-s',  'LineWidth',1.8,'MarkerSize',7); hold on;
semilogy(SNR_dB_vec, BER_gold, 'b-o',  'LineWidth',1.8,'MarkerSize',7);
semilogy(SNR_dB_vec, BER_mseq, 'r-^',  'LineWidth',1.8,'MarkerSize',7);
grid on;
xlabel('Transmit SNR [dB]','FontSize',13);
ylabel('BER','FontSize',13);
legend('Zadoff-Chu sequence','gold sequence','m-sequence',...
       'Location','southwest','FontSize',11);
xlim([13,20]); ylim([1e-5,1e0]);
set(gca,'FontSize',12,'YTick',[1e-5,1e-4,1e-3,1e-2,1e-1,1e0]);
title('Fig. 4: BER Performance','FontSize',13);

% %% =========================================================================
% %  PLOT FIG. 5 - RMSE
% %  [COMMENTED OUT — will fix after BER is finalized]
% %% =========================================================================
% figure('Position',[800,100,650,500],'Color','w');
% plot(SNR_dB_vec, RMSE_all(1,:), 'k-',   'LineWidth',1.8); hold on;
% plot(SNR_dB_vec, RMSE_all(2,:), 'k-o',  'LineWidth',1.8,'MarkerSize',8);
% plot(SNR_dB_vec, RMSE_all(3,:), 'k-x',  'LineWidth',1.8,'MarkerSize',10);
% grid on;
% xlabel('Transmit SNR [dB]','FontSize',13);
% ylabel('RMSE [m]','FontSize',13);
% legend('Bandwidth: 30.72 MHz','Bandwidth: 61.44 MHz','Bandwidth: 184.32 MHz',...
%        'Location','northeast','FontSize',11);
% xlim([13,20]); ylim([0,40]);
% set(gca,'FontSize',12);
% title('Fig. 5: RMSE Localization Performance','FontSize',13);

fprintf('\nSimulation complete.\n');

%% =========================================================================
%%  LOCAL FUNCTIONS
%% =========================================================================

%--------------------------------------------------------------------------
% Generate m-sequence of degree n from primitive polynomial
% poly_exponents: all exponents of the polynomial, e.g. [10,7] for x^10+x^7+1
% The constant term (+1) is always implicit.
% Recurrence: for x^n + x^a + 1, s(k) = s(k-(n-a)) + s(k-n)
% Register: reg(i) = s(k-i), so tap positions are (n-a) and n.
%--------------------------------------------------------------------------
function mseq = generate_mseq(n, poly_exponents)
    N   = 2^n - 1;
    reg = ones(1,n);
    mseq = zeros(1,N);

    % Convert polynomial exponents to register tap positions
    % Exclude x^n term; for remaining exponents e, position = n - e
    % Always include position n for the constant term (+1)
    exponents = poly_exponents(poly_exponents < n);
    tap_positions = n - exponents;         % intermediate terms
    tap_positions = [tap_positions, n];    % constant term (+1) -> reg(n)

    for k=1:N
        mseq(k) = reg(end);
        fb = mod(sum(reg(tap_positions)), 2);
        reg = [fb, reg(1:end-1)];
    end
    mseq = 2*mseq - 1;   % {0,1} -> {-1,+1}
end

%--------------------------------------------------------------------------
% Generate Q candidate sequences by cyclic shifting s0 by NCS (Eq.1)
%--------------------------------------------------------------------------
function S = make_candidates(s0, NCS, Q)
    Ns = length(s0);
    S  = zeros(Q, Ns);
    for i=0:Q-1
        S(i+1,:) = circshift(s0(:).', i*NCS);
    end
end

%--------------------------------------------------------------------------
% Generate tapped delay line channel (Eq.6) — no LOS
% ALL L paths have consecutive integer delays: tau0, tau0+1, ..., tau0+L-1
% First L_local: local scattering (angle near theta)
% Last L_interf: external interference (arbitrary angles, same delay structure)
%--------------------------------------------------------------------------
function [h_delay, tau_int] = gen_channel(d0, theta, lambda, r_ant, J,...
    L_local, L_interf, dtheta_max, alpha_pl, Ts, c_light, ~)

    L = L_local + L_interf;
    h_delay  = zeros(J, L);
    tau_int  = zeros(1, L);
    tau0 = round(d0 / (c_light * Ts));

    for l = 1:L
        tau_int(l) = tau0 + (l - 1);   % Eq.(6): consecutive delays
        dl = d0;                        % All paths ~ d0 distance

        if l <= L_local
            dtheta  = (2*rand()-1) * dtheta_max;
            theta_l = theta + dtheta;   % Local scattering near theta
        else
            theta_l = (rand()-0.5) * pi; % Interference: arbitrary angle
        end

        beta_l = (randn + 1j*randn) / sqrt(2);
        for jj = 1:J
            phase = -1j*2*pi/lambda * (jj-1) * r_ant * cos(theta_l);
            h_delay(jj, l) = beta_l * exp(phase) * sqrt(dl^(-alpha_pl));
        end
    end
end

%--------------------------------------------------------------------------
% OFDM Transmitter: s_i -> DFT(Ns) -> map to NFFT subcarriers -> IFFT -> CP
% Implements Eq.(3), (4), (5)
%--------------------------------------------------------------------------
function xCP = ofdm_tx(s_i, Ns, NFFT, NCP)
    S_freq     = fft(s_i(:).', Ns);              % Eq.(3)
    S_map      = zeros(1, NFFT);
    S_map(1:Ns) = S_freq;                        % Zero-pad to NFFT
    x          = ifft(S_map, NFFT);              % Eq.(4)
    xCP        = [x(NFFT-NCP+1:NFFT), x];       % Eq.(5): Insert CP
end

%--------------------------------------------------------------------------
% OFDM Receiver: remove CP -> FFT -> demapping -> IDFT -> y_j(n)
% Implements Eq.(8), (9)
%--------------------------------------------------------------------------
function y_all = ofdm_rx(r_sig, J, Ns, NFFT, NCP)
    y_all = zeros(J, Ns);
    for jj=1:J
        r_nocp = r_sig(jj, NCP+1 : NCP+NFFT);  % Remove CP
        R_freq = fft(r_nocp, NFFT);              % Eq.(8)
        R_sub  = R_freq(1:Ns);                   % Subcarrier demapping
        y_all(jj,:) = ifft(R_sub, Ns);           % Eq.(9): IDFT of size Ns
    end
end

%--------------------------------------------------------------------------
% Propagate signal through tapped delay line channel and add AWGN
%--------------------------------------------------------------------------
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

%--------------------------------------------------------------------------
% Sequence Detection & Demodulation — Eq.(11),(12),(13)
%--------------------------------------------------------------------------
function [i_hat, kappa_hat_all] = detect_seq(y_all, s0, NCS, J, Q, use_diversity)
    if nargin < 6, use_diversity = true; end
    Ns = length(s0);
    i_per_ant     = zeros(1,J);
    kappa_per_ant = zeros(1,J);

    S0_fft = fft(s0(:).', Ns);

    for jj=1:J
        % Eq.(11): Circular cross-correlation via FFT
        Yj   = fft(y_all(jj,:), Ns);
        R_j  = abs(ifft(Yj .* conj(S0_fft)));

        [~, idx]  = max(R_j);
        kappa_hat = idx - 1;   % 0-based

        % Eq.(12): i_hat_j = floor(kappa_hat / NCS)
        i_j = floor(kappa_hat / NCS);
        i_j = min(i_j, Q-1);

        i_per_ant(jj)     = i_j;
        kappa_per_ant(jj) = kappa_hat;
    end

    if use_diversity && J > 1
        % Spatial diversity: mode across all J antennas (for RMSE/direction)
        i_hat = mode(i_per_ant);
    else
        % Single antenna detection (for BER — no diversity)
        i_hat = i_per_ant(1);
    end
    kappa_hat_all = kappa_per_ant;
end

%--------------------------------------------------------------------------
% Distance and Direction Estimation — Eq.(14),(15),(16)
%--------------------------------------------------------------------------
function [d_est, theta_est] = estimate_dist_dir(y_all, s0, NCS, J, Ns,...
    kappa_hat_all, i_hat, NFFT, Ts, c_light, A, angle_grid, P)

    % Eq.(14): Distance estimation using first antenna
    kappa_j1   = kappa_hat_all(1);
    delay_samp = kappa_j1 - i_hat * NCS;
    delay_samp = max(delay_samp, 0);
    d_est      = c_light * Ts * (NFFT/Ns) * delay_samp;

    % Eq.(15): Despreading — y_des(j) = s_kappa^H * y_j
    kappa_use = i_hat * NCS;
    s_kappa   = circshift(s0(:).', kappa_use);

    y_des = zeros(J, 1);
    for jj=1:J
        y_des(jj) = conj(s_kappa) * y_all(jj,:).';   % s^H * y
    end

    % Eq.(16): Direction finding via spatial correlation
    corr_val   = abs(A' * y_des);
    [~, p_hat] = max(corr_val);
    theta_est  = angle_grid(p_hat);
end

%--------------------------------------------------------------------------
% DBSCAN Clustering — Algorithm 1 and Eq.(17)
%--------------------------------------------------------------------------
function [d0_hat, theta_hat, success] = clustering_localization(...
    y_multi, s0, NCS, J, Ns, i_hat, NFFT, NCP, Ts, c_light,...
    A, angle_grid, P, eta, Q)

    num_seq = size(y_multi,1)/J;

    % Algorithm 1: collect D = [d1, d2, d3] for NCS delay lags per sequence
    % kappa_ic in {i_hat*NCS, ..., i_hat*NCS+NCS-1}
    D   = zeros(NCS * num_seq, 3);
    row = 0;

    for seq=1:num_seq
        y_cur = y_multi((seq-1)*J+1 : seq*J, :);   % J x Ns

        for offset=0:NCS-1
            % Algorithm 1: kappa_ic = i_hat*NCS + offset
            kappa_ic = mod(i_hat * NCS + offset, Ns);
            s_kappa  = circshift(s0(:).', kappa_ic);

            % d1: distance (Eq.14)
            d1 = c_light * Ts * (NFFT/Ns) * offset;

            % d3: power — |s_kappa^H * y_1|^2
            d3 = abs(conj(s_kappa) * y_cur(1,:).')^2;

            % Despreading for direction (Eq.15): y_des(j) = s_kappa^H * y_j
            y_des = zeros(J,1);
            for jj=1:J
                y_des(jj) = conj(s_kappa) * y_cur(jj,:).';
            end

            % d2: angle (Eq.16)
            corr_val   = abs(A' * y_des);
            [~,p_hat]  = max(corr_val);
            d2         = angle_grid(p_hat);

            row      = row+1;
            D(row,:) = [d1, d2, d3];
        end
    end
    D = D(1:row,:);

    % Power threshold: exclude low-power points (normalized d3 < -eta)
    d3_mean = mean(D(:,3));
    d3_std  = std(D(:,3));
    if d3_std > 0
        D_norm3 = (D(:,3) - d3_mean) / d3_std;
    else
        D_norm3 = zeros(size(D,1),1);
    end
    D = D(D_norm3 >= -eta, :);

    if size(D,1) < 3
        d0_hat=NaN; theta_hat=NaN; success=false; return;
    end

    % Normalize all three features (zero mean, unit std)
    D_norm = zeros(size(D));
    for col=1:3
        mu = mean(D(:,col));
        sg = std(D(:,col));
        if sg > 0
            D_norm(:,col) = (D(:,col)-mu)/sg;
        end
    end

    % DBSCAN clustering
    epsilon = 1.0;
    zeta    = 3;

    try
        labels = dbscan_custom(D_norm, epsilon, zeta);
    catch
        d0_hat=NaN; theta_hat=NaN; success=false; return;
    end

    % Find cluster with most points (excluding noise label=0)
    unique_labels = unique(labels);
    unique_labels = unique_labels(unique_labels > 0);

    if isempty(unique_labels)
        d0_hat=NaN; theta_hat=NaN; success=false; return;
    end

    best_label = -1;
    best_count = 0;
    for lbl=unique_labels'
        cnt = sum(labels==lbl);
        if cnt > best_count
            best_count = cnt;
            best_label = lbl;
        end
    end

    D_bar = D(labels==best_label, :);

    % Eq.(17): d0_hat = min(d_bar_1), theta_hat = mean(d_bar_2)
    d0_hat    = min(D_bar(:,1));
    theta_hat = mean(D_bar(:,2));
    success   = true;
end

%--------------------------------------------------------------------------
% Simple DBSCAN implementation
%--------------------------------------------------------------------------
function labels = dbscan_custom(X, epsilon, min_pts)
    n      = size(X,1);
    labels = zeros(n,1);
    cluster_id = 0;

    for i=1:n
        if labels(i) ~= 0, continue; end
        neighbors = find_neighbors(X, i, epsilon);

        if length(neighbors) < min_pts
            labels(i) = -1;
            continue;
        end

        cluster_id = cluster_id + 1;
        labels(i)  = cluster_id;

        seed_set = neighbors;
        seed_set(seed_set==i) = [];

        while ~isempty(seed_set)
            q = seed_set(1);
            seed_set(1) = [];

            if labels(q) == -1
                labels(q) = cluster_id;
            end
            if labels(q) ~= 0, continue; end

            labels(q) = cluster_id;
            q_neighbors = find_neighbors(X, q, epsilon);
            if length(q_neighbors) >= min_pts
                seed_set = union(seed_set, q_neighbors);
            end
        end
    end
    labels(labels==-1) = 0;
end

function neighbors = find_neighbors(X, i, epsilon)
    dists = sqrt(sum((X - X(i,:)).^2, 2));
    neighbors = find(dists <= epsilon);
end

%--------------------------------------------------------------------------
% BER trial: one Monte Carlo iteration
%--------------------------------------------------------------------------
function num_errors = run_ber_trial(S_cand, s0, Ns, NCS,...
    NFFT, NCP, J, d0, theta, lambda, r_ant, alpha_pl,...
    L_local, L_interf, dtheta_max, Ts, c_light, SNR_lin,...
    i_tx, tx_bits, idx_to_bits, Q, q_bits)

    s_i = S_cand(i_tx+1,:);
    xCP = ofdm_tx(s_i, Ns, NFFT, NCP);

    % Noise variance: account for OFDM subcarrier ratio + CP overhead.
    % The signal occupies Ns out of NFFT subcarriers, and the CP adds
    % (NFFT+NCP)/NFFT overhead. Combined factor = (NFFT+NCP)/Ns.
    noise_var = (NFFT + NCP) / (Ns * SNR_lin);

    [h_delay, tau_int] = gen_channel(d0, theta, lambda, r_ant, J,...
        L_local, L_interf, dtheta_max, alpha_pl, Ts, c_light, NCP);

    r_sig = apply_channel(xCP, h_delay, tau_int, J, L_local+L_interf,...
        NFFT, NCP, noise_var);

    y_all = ofdm_rx(r_sig, J, Ns, NFFT, NCP);

    % Single-antenna detection (all antennas see same |h| so mode=no gain)
    [i_hat, ~] = detect_seq(y_all, s0, NCS, J, Q, false);
    i_hat = max(0, min(i_hat, Q-1));

    rx_bits = idx_to_bits(i_hat+1,:);
    num_errors = sum(rx_bits ~= tx_bits);
end

%--------------------------------------------------------------------------
% RMSE trial: one Monte Carlo iteration (Steps 2+3)
%--------------------------------------------------------------------------
function [d0_hat, theta_hat, success] = run_rmse_trial(...
    S_cand, s0, Ns, NCS,...
    NFFT, NCP, J, d0, theta, lambda, r_ant, alpha_pl,...
    L_local, L_interf, dtheta_max, Ts, c_light, SNR_lin,...
    i_tx, Q, A, angle_grid, num_seq, eta)

    L = L_local + L_interf;

    % Generate channel ONCE (static during 5 sequences, ~137us)
    [h_delay, tau_int] = gen_channel(d0, theta, lambda, r_ant, J,...
        L_local, L_interf, dtheta_max, alpha_pl, Ts, c_light, NCP);

    % Collect received signals from num_seq sequences (Algorithm 1)
    y_multi  = zeros(num_seq*J, Ns);
    i_tx_cur = i_tx;

    for seq=1:num_seq
        s_i = S_cand(i_tx_cur+1,:);
        xCP = ofdm_tx(s_i, Ns, NFFT, NCP);

        % Same noise model as BER trial
        noise_var = (NFFT + NCP) / (Ns * SNR_lin);

        r_sig = apply_channel(xCP, h_delay, tau_int, J, L, NFFT, NCP, noise_var);
        y_all = ofdm_rx(r_sig, J, Ns, NFFT, NCP);
        y_multi((seq-1)*J+1:seq*J,:) = y_all;

        i_tx_cur = mod(i_tx_cur+1, Q);
    end

    % Detect sequence index from first received signal
    y_first = y_multi(1:J,:);
    [i_hat, ~] = detect_seq(y_first, s0, NCS, J, Q);

    % DBSCAN clustering localization
    P_grid = size(A,2);
    [d0_hat, theta_hat, success] = clustering_localization(...
        y_multi, s0, NCS, J, Ns, i_hat, NFFT, NCP, Ts, c_light,...
        A, angle_grid, P_grid, eta, Q);
end