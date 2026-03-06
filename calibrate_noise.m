% Calibration script for OFDM-ISAC noise variance
clc; clear; close all;

% Run parameters
d0 = 40; theta_true = 45*pi/180; fc = 1.8e9; c_light = 3e8; lambda = c_light/fc;
NFFT = 2048; NCP = 256; J = 16; r_ant = lambda/2; alpha_pl = 2;
L = 10; dtheta_max = 10*pi/180;
L_interf = round(0.2*L); L_local = L - L_interf;
q_bits = 6; Q = 2^q_bits;
Ns_ZC = 839; NCS_ZC = 13;
Ts = 1/30.72e6;

% generate sequence
s0_ZC = exp(-1j * pi * 1 * (0:Ns_ZC-1) .* (1:Ns_ZC) / Ns_ZC);
S_ZC = zeros(Q, Ns_ZC);
for q=1:Q
    shift_val = (q-1) * NCS_ZC;
    S_ZC(q,:) = circshift(s0_ZC, shift_val);
end
idx_to_bits = dec2bin(0:Q-1, q_bits) - '0';

SNR_dB = 13;
SNR_lin = 10^(SNR_dB/10);
num_trials = 2000;

scalars = logspace(-1, 2, 12);
ber_results = zeros(1, length(scalars));

fprintf('Calibrating noise scalar at 13dB with mode voting...\n');

for idx = 1:length(scalars)
    scalar = scalars(idx);
    noise_var = scalar * (NFFT + NCP) / (Ns_ZC * SNR_lin);
    
    total_err = 0;
    for trial = 1:num_trials
        i_tx = randi([0,Q-1]);
        tx_bits = idx_to_bits(i_tx+1,:);
        s_i = S_ZC(i_tx+1,:);
        
        % TX (identical to main script)
        S_freq = fft(s_i(:).', Ns_ZC);
        S_map = zeros(1, NFFT);
        S_map(1:Ns_ZC) = S_freq;
        x = ifft(S_map, NFFT);
        xCP = [x(NFFT-NCP+1:NFFT), x];

        % Channel
        L_tot = L_local + L_interf;
        h_delay = zeros(J, L_tot);
        tau_int = zeros(1, L_tot);
        tau0 = round(d0 / (c_light * Ts));
        for l = 1:L_tot
            tau_int(l) = tau0 + (l - 1);
            if l <= L_local
                theta_l = theta_true + (2*rand()-1)*dtheta_max;
            else
                theta_l = (rand()-0.5)*pi;
            end
            beta_l = (randn + 1j*randn)/sqrt(2);
            for jj = 1:J
                ph = -1j*2*pi/lambda * (jj-1)*r_ant*cos(theta_l);
                h_delay(jj, l) = beta_l * exp(ph) * sqrt(d0^(-alpha_pl));
            end
        end

        % Apply channel
        sig_len = NFFT+NCP;
        r_sig = zeros(J, sig_len);
        for jj=1:J
            r_j = zeros(1, sig_len);
            for l=1:L_tot
                tau_l = tau_int(l);
                if tau_l < sig_len
                    xCP_del = [zeros(1,tau_l), xCP(1:sig_len-tau_l)];
                    r_j = r_j + h_delay(jj,l) * xCP_del;
                end
            end
            noise = sqrt(noise_var/2)*(randn(1,sig_len)+1j*randn(1,sig_len));
            r_sig(jj,:) = r_j + noise;
        end

        % RX
        y_all = zeros(J, Ns_ZC);
        for jj=1:J
            r_nocp = r_sig(jj, NCP+1 : NCP+NFFT); 
            R_freq = fft(r_nocp, NFFT);
            R_sub = R_freq(1:Ns_ZC);
            y_all(jj,:) = ifft(R_sub, Ns_ZC);
        end

        % Detect (mode voting)
        i_per_ant = zeros(1,J);
        S0_fft = fft(s0_ZC(:).', Ns_ZC);
        for jj=1:J
            Yj = fft(y_all(jj,:), Ns_ZC);
            R_j = abs(ifft(Yj .* conj(S0_fft)));
            [~, peak_idx] = max(R_j);
            kappa_hat = peak_idx - 1;
            i_j = floor(kappa_hat / NCS_ZC);
            i_per_ant(jj) = min(i_j, Q-1);
        end
        
        % Mode over 16 antennas
        i_hat = mode(i_per_ant);
        i_hat = max(0, min(i_hat, Q-1));
        
        rx_bits = idx_to_bits(i_hat+1,:);
        total_err = total_err + sum(rx_bits ~= tx_bits);
    end
    
    ber = total_err / (num_trials * q_bits);
    ber_results(idx) = ber;
    fprintf('Scalar = %7.2f -> BER = %.5f\n', scalar, ber);
end
