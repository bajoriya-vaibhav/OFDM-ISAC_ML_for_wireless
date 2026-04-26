"""
Complete OFDM-ISAC System Pipeline
Integrates TX/RX, channel, detection, and clustering
Reference: Paper Sec. II-III and MATLAB run_ber_trial()
"""

import numpy as np
from system_params import *
from orthogonal_sequences import *
from channel_noise import *
from ofdm_modem import *
from sequence_detection import *
from dbscan_clustering import *


class OFDMISACSystem:
    """
    Complete OFDM-ISAC system implementation
    """
    
    def __init__(self, seq_type='ZC', BW=30.72e6, zc_root=None,
                 ue_d0=None, ue_theta=None):
        """
        Initialize OFDM-ISAC system
        
        Args:
            seq_type: 'ZC', 'mseq', or 'gold'
            BW: System bandwidth [Hz]
            zc_root: (optional) Custom ZC root index for multi-user mode.
                     If None, uses default root=1 from initialize_sequences().
            ue_d0: (optional) UE distance [m]. Defaults to system_params.d0.
            ue_theta: (optional) UE angle [rad]. Defaults to system_params.theta_true.
        """
        # Initialize sequences — use per-root dict when zc_root is specified
        if zc_root is not None and seq_type == 'ZC':
            from orthogonal_sequences import initialize_sequences_for_root
            self.seq_dict = initialize_sequences_for_root(zc_root)
        else:
            self.seq_dict = initialize_sequences()
        self.seq_type = seq_type
        self.seq_info = self.seq_dict[seq_type]
        
        # OFDM parameters
        self.NFFT = NFFT
        self.NCP = NCP
        self.Ns = self.seq_info['Ns']
        self.NCS = self.seq_info['NCS']
        self.BW = BW
        self.Ts = 1 / BW  # Sampling time
        
        # Per-UE physical parameters (defaults to global params)
        self.ue_d0 = ue_d0 if ue_d0 is not None else d0
        self.ue_theta = ue_theta if ue_theta is not None else theta_true
        
        # Gray code
        self.gray_enc, self.idx_to_bits = generate_gray_code(q_bits)
        
        # Components
        self.tx = OFDMTransmitter(NFFT=self.NFFT, NCP=self.NCP)
        self.rx = OFDMReceiver(NFFT=self.NFFT, NCP=self.NCP)
        self.channel = ChannelModel(d0=self.ue_d0, theta=self.ue_theta)
        self.detector = SequenceDetector(J=J)
        self.localizer = LocalizationEstimator(J=J)
        self.clustering = ClusteringLocalizationEngine(
            NCS=self.NCS, J=J, lambda_wave=lambda_wave, r_ant=r_ant,
            eta_threshold=eta_threshold, epsilon=epsilon, min_samples=min_samples
        )
    
    def transmit(self, i_tx):
        """
        Transmit sequence index as OFDM signal
        
        Args:
            i_tx: Sequence index to transmit
        
        Returns:
            x_CP: OFDM signal with CP
            s_i: Transmitted sequence
        """
        s_i = self.seq_info['S'][i_tx, :]
        x_CP = self.tx.modulate(s_i, self.Ns)
        return x_CP, s_i
    
    def receive_channel(self, x_CP, SNR_dB, apply_channel=True):
        """
        Apply channel and noise
        
        Args:
            x_CP: Transmitted OFDM signal
            SNR_dB: Signal-to-noise ratio [dB]
            apply_channel: Whether to apply multipath channel
        
        Returns:
            r_sig: Received signal (J, N)
            h_delay: Channel coefficients
            tau_int: Delay indices
        """
        SNR_lin = snr_db_to_linear(SNR_dB)
        noise_var = NoiseModel.calculate_noise_variance(SNR_lin, self.NFFT, 
                                                         self.NCP, self.Ns)
        
        # Generate channel
        h_delay, tau_int = self.channel.generate_channel(self.Ts, c_light=c_light)
        
        # Apply channel
        L = self.channel.L
        r_sig = self._apply_channel_and_noise(x_CP, h_delay, tau_int, L, 
                                             noise_var, self.NFFT, self.NCP)
        
        return r_sig, h_delay, tau_int
    
    def _apply_channel_and_noise(self, x_CP, h_delay, tau_int, L, noise_var, 
                                NFFT, NCP):
        """
        Apply multipath channel and AWGN
        
        Implements Eq. (6) in paper
        
        Args:
            x_CP: Transmitted signal
            h_delay: Channel coefficients (J, L)
            tau_int: Delay indices (L,)
            L: Number of paths
            noise_var: Noise variance
            NFFT: FFT size
            NCP: CP length
        
        Returns:
            r_sig: Received signal (J, NFFT + NCP)
        """
        sig_len = NFFT + NCP
        r_sig = np.zeros((J, sig_len), dtype=complex)
        
        for jj in range(J):
            r_j = np.zeros(sig_len, dtype=complex)
            
            # Sum contributions from all paths
            for l in range(L):
                tau_l = tau_int[l]
                if tau_l < sig_len:
                    # Delay the signal
                    x_delayed = np.concatenate([np.zeros(tau_l), x_CP[:sig_len - tau_l]])
                    r_j = r_j + h_delay[jj, l] * x_delayed
            
            # Add AWGN
            noise = np.sqrt(noise_var / 2) * (np.random.randn(sig_len) + 
                                             1j * np.random.randn(sig_len))
            r_sig[jj, :] = r_j + noise
        
        return r_sig
    
    def receive_demod(self, r_sig):
        """
        Receive and demodulate OFDM
        
        Args:
            r_sig: Received signal (J, N)
        
        Returns:
            y_all: Received sequences (J, Ns)
        """
        y_all = self.rx.demodulate(r_sig, self.Ns, J=J)
        return y_all
    
    def detect_and_demodulate(self, y_all):
        """
        Detect sequence and demodulate bits
        
        Args:
            y_all: Received sequences (J, Ns)
        
        Returns:
            i_hat: Detected sequence index
            b_hat: Detected bits
            kappa_hat_all: Peak indices for all antennas
        """
        i_hat, kappa_hat_all = self.detector.detect_sequence(
            y_all, self.seq_info['s0'], self.NCS, Q, use_diversity=True
        )
        
        _, b_hat = self.detector.demodulate(i_hat, self.gray_enc, self.idx_to_bits)
        
        return i_hat, b_hat, kappa_hat_all
    
    def estimate_localization(self, y_all, i_hat, kappa_hat_all, 
                             use_clustering=True, angle_grid_size=181):
        """
        Estimate distance and direction (with or without clustering)
        
        Args:
            y_all: Received sequences (J, Ns)
            i_hat: Detected sequence index
            kappa_hat_all: Peak indices for all antennas
            use_clustering: Use DBSCAN clustering for improved resolution
            angle_grid_size: Grid size for angle estimation
        
        Returns:
            d_hat: Estimated distance
            theta_hat: Estimated direction
            cluster_data: Cluster information (if using clustering)
        """
        if use_clustering:
            d_hat, theta_hat, cluster_data = self.clustering.estimate_localization(
                y_all, self.seq_info['s0'], i_hat, kappa_hat_all,
                self.NFFT, self.Ns, self.Ts, angle_grid_size
            )
        else:
            # Simple estimation without clustering
            d_est = self.localizer.estimate_distance(kappa_hat_all, i_hat, 
                                                      self.NCS, self.NFFT, 
                                                      self.Ns, self.Ts)
            theta_est = self.localizer.estimate_direction_beamforming(
                y_all, self.seq_info['s0'], kappa_hat_all, i_hat, self.NCS,
                angle_grid_size
            )
            
            # Use median/mean
            d_hat = np.median(d_est)
            theta_hat = theta_est
            cluster_data = None
        
        return d_hat, theta_hat, cluster_data
    
    def run_ber_trial(self, i_tx, SNR_dB):
        """
        Run single BER trial
        
        Args:
            i_tx: Sequence index to transmit
            SNR_dB: SNR in dB
        
        Returns:
            num_bit_errors: Number of bit errors
            tx_bits: Transmitted bits
            rx_bits: Received bits
        """
        # Transmit
        x_CP, s_i = self.transmit(i_tx)
        tx_bits = self.idx_to_bits[i_tx, :]
        
        # Channel and receive
        r_sig, _, _ = self.receive_channel(x_CP, SNR_dB, apply_channel=True)
        y_all = self.receive_demod(r_sig)
        
        # Detect
        i_hat, rx_bits, _ = self.detect_and_demodulate(y_all)
        
        # Calculate errors
        num_bit_errors = np.sum(rx_bits != tx_bits)
        
        return num_bit_errors, tx_bits, rx_bits
    
    def run_rmse_trial(self, i_tx, SNR_dB, n_sequences=5):
        """
        Single RMSE Monte-Carlo trial — paper Sec. III.B and Algorithm 1.

        Steps:
          1. Sensing noise: noise_var = 1/SNR_lin  (no BER calibration factor)
          2. Generate one channel realisation (static across all M symbols)
          3. Transmit M consecutive symbols, receive on all J antennas
          4. Feed stacked received matrix to ClusteringLocalizationEngine
          5. If clustering fails → direct estimation fallback (Eq 14, 16)
        """
        d_true_val     = float(self.ue_d0)
        theta_true_val = float(self.ue_theta)
        tx_bits       = self.idx_to_bits[i_tx, :]

        # ── 1. Unified noise ────────────────────────────────────────────────
        SNR_lin  = snr_db_to_linear(SNR_dB)
        noise_var = NoiseModel.calculate_noise_variance(SNR_lin, self.NFFT, self.NCP, self.Ns)

        # ── 2. Collect M sequential symbols ─────────────────────────────────
        y_multi  = np.zeros((n_sequences * J, self.Ns), dtype=complex)
        i_tx_cur = i_tx
        
        # The paper specifies: "parameters of individual path vary with time slot m 
        # following standard continuous-time Rayleigh fading process."
        for m in range(n_sequences):
            h_delay, tau_int = self.channel.generate_channel(self.Ts, c_light=c_light)
            L_ch = self.channel.L
            
            x_CP, _ = self.transmit(i_tx_cur)
            r_sig   = self._apply_channel_and_noise(x_CP, h_delay, tau_int,
                                                     L_ch, noise_var,
                                                     self.NFFT, self.NCP)
            y_all   = self.receive_demod(r_sig)
            y_multi[m * J:(m + 1) * J, :] = y_all
            i_tx_cur = (i_tx_cur + 1) % Q

        # ── 4a. Detect sequence from FIRST symbol ────────────────────────────
        y_first = y_multi[0:J, :]
        i_hat, kappa_hat_all = self.detector.detect_sequence(
            y_first, self.seq_info['s0'], self.NCS, Q, use_diversity=True
        )
        _, rx_bits = self.detector.demodulate(i_hat, self.gray_enc, self.idx_to_bits)

        # ── 4b. Clustering (Algorithm 1 + Eq 17) ────────────────────────────
        d_hat, theta_hat, _ = self.clustering.estimate_localization(
            y_multi, self.seq_info['s0'], i_hat, kappa_hat_all,
            self.NFFT, self.Ns, self.Ts, Q,
            num_seq=n_sequences, angle_grid_size=361
        )

        # ── 5. Fallback: direct estimation if clustering failed ──────────────
        if d_hat is None or d_hat <= 0:
            # Fallback: if DBSCAN failed, just rely on the single highest-power point
            D_raw = self.clustering.collect_data(
                y_multi, self.seq_info['s0'], i_hat,
                self.NFFT, self.Ns, self.Ts, Q,
                num_seq=n_sequences, angle_grid_size=361
            )
            # Find the row with maximum power (d3)
            best_idx = np.argmax(D_raw[:, 2])
            d_hat = float(D_raw[best_idx, 0])
            theta_hat = float(D_raw[best_idx, 1])

        return d_hat, theta_hat, d_true_val, theta_true_val, tx_bits, rx_bits


if __name__ == "__main__":
    print("Testing OFDM-ISAC System...")
    
    # Create system for Zadoff-Chu sequence
    system = OFDMISACSystem(seq_type='ZC', BW=30.72e6)
    
    print(f"\nSystem Configuration:")
    print(f"  Sequence type: {system.seq_type}")
    print(f"  Sequence length (Ns): {system.Ns}")
    print(f"  Cyclic shift (NCS): {system.NCS}")
    print(f"  Bandwidth: {system.BW / 1e6:.2f} MHz")
    print(f"  Number of antennas: {J}")
    
    # Test BER trial
    print(f"\nTesting BER Trial...")
    i_tx = 10  # Transmit sequence index 10
    SNR_dB = 15
    
    num_errors, tx_bits, rx_bits = system.run_ber_trial(i_tx, SNR_dB)
    print(f"  Transmitted sequence index: {i_tx}")
    print(f"  SNR: {SNR_dB} dB")
    print(f"  TX bits: {tx_bits}")
    print(f"  RX bits: {rx_bits}")
    print(f"  Bit errors: {num_errors}/{len(tx_bits)}")
    
    # Test RMSE trial
    print(f"\nTesting RMSE Trial...")
    d_hat, theta_hat, d_true, theta_true_val, tx_bits, rx_bits = \
        system.run_rmse_trial(i_tx, SNR_dB, n_sequences=5)
    
    print(f"  True distance: {d_true:.2f} m")
    print(f"  Estimated distance: {d_hat:.2f} m")
    print(f"  True angle: {theta_true_val * 180 / np.pi:.1f}°")
    if theta_hat is not None:
        print(f"  Estimated angle: {theta_hat * 180 / np.pi:.1f}°")
    else:
        print(f"  Estimated angle: Failed (None)")
