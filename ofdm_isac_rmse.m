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
eta           = 1;              % Power threshold for DBSCAN

% 20% of taps are external interference
L_interf      = round(0.2*L);   % = 2 interference taps
L_local       = L - L_interf;   % = 8 locally scattered taps

% Index modulation: 6 bits
q_bits        = 6;
Q             = 2^q_bits;       % 64 candidate sequences

% ZC: Ns=839, NCS=13
Ns_ZC         = 839;   NCS_ZC  = 13;

% Angle search grid (P angles)
P             = 361;
angle_grid    = linspace(-pi/2, pi/2, P);  % [-90, 90] degrees

% SNR range
SNR_dB_vec    = 13:1:20;
num_snr       = length(SNR_dB_vec);

% Monte Carlo trials
num_trials_RMSE = 1000;

% DBSCAN uses 5*NCS data samples (5 sequences)
num_seq_dbscan  = 5;

fprintf('OFDM-ISAC RMSE Sensing Simulation Starting...\n');
fprintf('Parameters: d0=%dm, theta=%ddeg, J=%d antennas, L=%d paths\n',...
    d0, round(theta_true*180/pi), J, L);

%% Orthogonal sequence generation (ZC sequence only for RMSE)

% Zadoff-Chu sequence (root u=1)
u = 1;
n_zc = (0:Ns_ZC-1)';
s0_ZC = exp(-1j*pi*u*n_zc.*(n_zc+1)/Ns_ZC);

% Generate Q candidate sequences by cyclic shift (Eq. 1)
S_ZC   = make_candidates(s0_ZC,   NCS_ZC,  Q);

%% Steering vector matrix A (J x P) for direction finding (Eq. 16)
A = zeros(J,P);
for p=1:P
    for jj=1:J
        A(jj,p) = exp(-1j*2*pi/lambda*(jj-1)*r_ant*cos(angle_grid(p)));
    end
end

%% =========================================================================
%  FIG. 5: RMSE SIMULATION (ZC sequence, three bandwidths)
%% =========================================================================
fprintf('\n--- Fig.5: RMSE Simulation (ZC sequence, 3 bandwidths) ---\n');
BW_vec    = [30.72e6, 61.44e6, 184.32e6];
num_BW    = length(BW_vec);
RMSE_all  = zeros(num_BW, num_snr);

for bw_idx = 1:num_BW
    BW_cur = BW_vec(bw_idx);
    Ts_cur = 1/BW_cur;
    fprintf('\nBandwidth = %.2f MHz\n', BW_cur/1e6);

    for si = 1:num_snr
        SNR_dB  = SNR_dB_vec(si);
        SNR_lin = 10^(SNR_dB/10);
        rmse_sq_sum = 0;
        valid_count = 0;

        for trial = 1:num_trials_RMSE
            i_tx = randi([0,Q-1]);

            [d0_hat, theta_hat, success] = run_rmse_trial(...
                S_ZC, s0_ZC, Ns_ZC, NCS_ZC,...
                NFFT, NCP, J, d0, theta_true, lambda, r_ant, alpha_pl,...
                L_local, L_interf, dtheta_max, Ts_cur, c_light, SNR_lin,...
                i_tx, Q, A, angle_grid, num_seq_dbscan, eta);

            if success
                err_x = d0_hat*cos(theta_hat) - d0*cos(theta_true);
                err_y = d0_hat*sin(theta_hat) - d0*sin(theta_true);
                rmse_sq_sum = rmse_sq_sum + err_x^2 + err_y^2;
                valid_count = valid_count + 1;
            end
        end

        if valid_count > 0
            RMSE_all(bw_idx,si) = sqrt(rmse_sq_sum / valid_count);
        else
            RMSE_all(bw_idx,si) = NaN;
        end

        fprintf('  SNR=%2ddB | RMSE=%.2f m (valid=%d/%d)\n',...
            SNR_dB, RMSE_all(bw_idx,si), valid_count, num_trials_RMSE);
    end
end

%% =========================================================================
%  PLOT FIG. 5 - RMSE
%% =========================================================================
figure('Position',[800,100,650,500],'Color','w');
plot(SNR_dB_vec, RMSE_all(1,:), 'k-',   'LineWidth',1.8); hold on;
plot(SNR_dB_vec, RMSE_all(2,:), 'k-o',  'LineWidth',1.8,'MarkerSize',8);
plot(SNR_dB_vec, RMSE_all(3,:), 'k-x',  'LineWidth',1.8,'MarkerSize',10);
grid on;
grid minor;
set(gca, 'GridColor', [0.15 0.15 0.15], 'GridAlpha', 0.5);
set(gca, 'MinorGridColor', [0.15 0.15 0.15], 'MinorGridAlpha', 0.2);
xlabel('Transmit SNR [dB]','FontSize',13);
ylabel('RMSE [m]','FontSize',13);
legend('Bandwidth: 30.72 MHz','Bandwidth: 61.44 MHz','Bandwidth: 184.32 MHz',...
       'Location','northeast','FontSize',11);
xlim([13,20]); ylim([0,40]);
set(gca,'FontSize',12, 'Color', 'w');
set(gcf, 'Color', 'w');
title('Fig. 5: RMSE Localization Performance','FontSize',13);

fprintf('\nRMSE Simulation complete.\n');

%% =========================================================================
%% LOCAL FUNCTIONS
%% =========================================================================

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

function [d0_hat, theta_hat, success] = clustering_localization(...
    y_multi, s0, NCS, J, Ns, i_hat, NFFT, NCP, Ts, c_light,...
    A, angle_grid, P, eta, Q)

    num_seq = size(y_multi,1)/J;
    D   = zeros(NCS * num_seq, 3);
    row = 0;

    for seq=1:num_seq
        y_cur = y_multi((seq-1)*J+1 : seq*J, :);   
        for offset=0:NCS-1
            kappa_ic = mod(i_hat * NCS + offset, Ns);
            s_kappa  = circshift(s0(:).', kappa_ic);
            
            d1 = c_light * Ts * (NFFT/Ns) * offset;
            d3 = abs(conj(s_kappa) * y_cur(1,:).')^2;
            
            y_des = zeros(J,1);
            for jj=1:J
                y_des(jj) = conj(s_kappa) * y_cur(jj,:).';
            end
            
            corr_val   = abs(A' * y_des);
            [~,p_hat]  = max(corr_val);
            d2         = angle_grid(p_hat);

            row      = row+1;
            D(row,:) = [d1, d2, d3];
        end
    end
    D = D(1:row,:);

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

    D_norm = zeros(size(D));
    for col=1:3
        mu = mean(D(:,col));
        sg = std(D(:,col));
        if sg > 0
            D_norm(:,col) = (D(:,col)-mu)/sg;
        end
    end

    epsilon = 1.0;
    zeta    = 3;

    try
        labels = dbscan_custom(D_norm, epsilon, zeta);
    catch
        d0_hat=NaN; theta_hat=NaN; success=false; return;
    end

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
    d0_hat    = min(D_bar(:,1));
    theta_hat = mean(D_bar(:,2));
    success   = true;
end

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

function [d0_hat, theta_hat, success] = run_rmse_trial(...
    S_cand, s0, Ns, NCS,...
    NFFT, NCP, J, d0, theta, lambda, r_ant, alpha_pl,...
    L_local, L_interf, dtheta_max, Ts, c_light, SNR_lin,...
    i_tx, Q, A, angle_grid, num_seq, eta)

    L = L_local + L_interf;

    [h_delay, tau_int] = gen_channel(d0, theta, lambda, r_ant, J,...
        L_local, L_interf, dtheta_max, alpha_pl, Ts, c_light, NCP);

    y_multi  = zeros(num_seq*J, Ns);
    i_tx_cur = i_tx;

    for seq=1:num_seq
        s_i = S_cand(i_tx_cur+1,:);
        xCP = ofdm_tx(s_i, Ns, NFFT, NCP);

        noise_var = (NFFT + NCP) / (Ns * SNR_lin);

        r_sig = apply_channel(xCP, h_delay, tau_int, J, L, NFFT, NCP, noise_var);
        y_all = ofdm_rx(r_sig, J, Ns, NFFT, NCP);
        y_multi((seq-1)*J+1:seq*J,:) = y_all;

        i_tx_cur = mod(i_tx_cur+1, Q);
    end

    y_first = y_multi(1:J,:);
    [i_hat, ~] = detect_seq(y_first, s0, NCS, J, Q);

    P_grid = size(A,2);
    [d0_hat, theta_hat, success] = clustering_localization(...
        y_multi, s0, NCS, J, Ns, i_hat, NFFT, NCP, Ts, c_light,...
        A, angle_grid, P_grid, eta, Q);
end
