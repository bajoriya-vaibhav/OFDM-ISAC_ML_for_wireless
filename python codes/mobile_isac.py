"""
Mobile UE OFDM-ISAC System
Extends the static OFDMISACSystem with Doppler-aware channel and velocity estimation.
Single-user mobile implementation.
"""

import numpy as np
from system_params import *
from orthogonal_sequences import *
from channel_noise import *
from ofdm_modem import *
from sequence_detection import *
from dbscan_clustering import *
from ofdm_isac import OFDMISACSystem


class MobileOFDMISACSystem(OFDMISACSystem):
    """
    Mobile UE extension of the static OFDM-ISAC system.
    
    Key differences from static system:
      - Channel uses Doppler phase rotation per symbol
      - Uses M=32 slow-time symbols for Doppler FFT
      - DBSCAN operates on 4D features: [distance, angle, velocity, power]
      - Returns velocity estimate alongside distance and angle
    """

    def __init__(self, seq_type='ZC', BW=30.72e6, ue_v=None, ue_d0=None, ue_theta=None):
        """
        Initialize mobile ISAC system.

        Args:
            seq_type: Sequence type ('ZC', 'mseq', 'gold')
            BW: System bandwidth [Hz]
            ue_v: UE velocity [m/s]. Defaults to system_params.v_ue.
            ue_d0: UE distance [m]. Defaults to system_params.d0.
            ue_theta: UE angle [rad]. Defaults to system_params.theta_true.
        """
        super().__init__(seq_type=seq_type, BW=BW, ue_d0=ue_d0, ue_theta=ue_theta)

        self.ue_v = ue_v if ue_v is not None else v_ue
        self.T_sym = (NFFT + NCP) / BW  # Symbol duration for this BW
        self.M = M_doppler               # Slow-time symbol count

    def run_ber_trial(self, i_tx, SNR_dB):
        """
        BER trial with Doppler channel.

        Doppler adds a phase rotation to the channel but does NOT fundamentally
        change the BER performance for a single OFDM symbol (the CP absorbs
        the delay and the phase is constant within one symbol).
        We use the Doppler channel at m=0 for a fair comparison.
        """
        # Transmit
        x_CP, s_i = self.transmit(i_tx)
        tx_bits = self.idx_to_bits[i_tx, :]

        # Channel with Doppler at m=0
        SNR_lin = snr_db_to_linear(SNR_dB)
        noise_var = NoiseModel.calculate_noise_variance(SNR_lin, self.NFFT,
                                                         self.NCP, self.Ns)
        self.channel.reset_doppler_state()
        h_delay, tau_int = self.channel.generate_channel_doppler(
            self.Ts, self.ue_v, m_symbol=0, T_sym=self.T_sym)
        L_ch = self.channel.L

        r_sig = self._apply_channel_and_noise(x_CP, h_delay, tau_int,
                                               L_ch, noise_var,
                                               self.NFFT, self.NCP)
        y_all = self.receive_demod(r_sig)

        # Detect
        i_hat, rx_bits, _ = self.detect_and_demodulate(y_all)

        # Errors
        num_bit_errors = np.sum(rx_bits != tx_bits)
        return num_bit_errors, tx_bits, rx_bits

    def run_rmse_trial(self, i_tx, SNR_dB, n_sequences=None):
        """
        RMSE trial with Doppler-aware channel and velocity estimation.

        Steps:
          1. Generate base channel once (cached via Doppler state)
          2. For each of M symbols, apply Doppler phase e^{j2πf_D·m·T_sym}
          3. Collect M symbols → slow-time FFT for Doppler/velocity
          4. DBSCAN on 4D features [dist, angle, velocity, power]

        Returns:
            d_hat, theta_hat, v_hat, d_true, theta_true, v_true,
            tx_bits, rx_bits
        """
        if n_sequences is None:
            n_sequences = self.M

        d_true_val = float(self.ue_d0)
        theta_true_val = float(self.ue_theta)
        v_true_val = float(self.ue_v)
        tx_bits = self.idx_to_bits[i_tx, :]

        # True Doppler frequency for reference
        f_D_true = self.ue_v * np.cos(self.ue_theta) / lambda_wave

        # Noise
        SNR_lin = snr_db_to_linear(SNR_dB)
        noise_var = NoiseModel.calculate_noise_variance(SNR_lin, self.NFFT,
                                                         self.NCP, self.Ns)

        # Reset Doppler state for fresh base channel
        self.channel.reset_doppler_state()

        # Collect M sequential symbols with Doppler channel
        y_multi = np.zeros((n_sequences * J, self.Ns), dtype=complex)
        i_tx_cur = i_tx

        for m in range(n_sequences):
            h_delay, tau_int = self.channel.generate_channel_doppler(
                self.Ts, self.ue_v, m_symbol=m, T_sym=self.T_sym)
            L_ch = self.channel.L

            x_CP, _ = self.transmit(i_tx_cur)
            r_sig = self._apply_channel_and_noise(x_CP, h_delay, tau_int,
                                                    L_ch, noise_var,
                                                    self.NFFT, self.NCP)
            y_all = self.receive_demod(r_sig)
            y_multi[m * J:(m + 1) * J, :] = y_all
            i_tx_cur = (i_tx_cur + 1) % Q

        # Detect from first symbol
        y_first = y_multi[0:J, :]
        i_hat, kappa_hat_all = self.detector.detect_sequence(
            y_first, self.seq_info['s0'], self.NCS, Q, use_diversity=True)
        _, rx_bits = self.detector.demodulate(i_hat, self.gray_enc,
                                               self.idx_to_bits)

        # Doppler-aware clustering
        d_hat, theta_hat, v_hat, cl_data = \
            self.clustering.estimate_localization_doppler(
                y_multi, self.seq_info['s0'], i_hat, kappa_hat_all,
                self.NFFT, self.Ns, self.Ts, Q,
                T_sym=self.T_sym, lambda_wave=lambda_wave,
                num_seq=n_sequences, angle_grid_size=361)

        # Fallback if clustering fails
        if d_hat is None or d_hat <= 0:
            D_raw = self.clustering.collect_data_doppler(
                y_multi, self.seq_info['s0'], i_hat,
                self.NFFT, self.Ns, self.Ts, Q,
                T_sym=self.T_sym, lambda_wave=lambda_wave,
                num_seq=n_sequences, angle_grid_size=361)
            best_idx = np.argmax(D_raw[:, 3])
            d_hat = float(D_raw[best_idx, 0])
            theta_hat = float(D_raw[best_idx, 1])
            v_hat = float(D_raw[best_idx, 2])

        return (d_hat, theta_hat, v_hat,
                d_true_val, theta_true_val, v_true_val,
                tx_bits, rx_bits)


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Mobile UE OFDM-ISAC System Test")
    print("=" * 60)

    system = MobileOFDMISACSystem(seq_type='ZC', BW=30.72e6)

    print(f"  Distance: {system.ue_d0:.1f} m")
    print(f"  Angle: {system.ue_theta * 180 / np.pi:.1f}deg")
    print(f"  Velocity: {system.ue_v:.1f} m/s ({system.ue_v * 3.6:.0f} km/h)")
    print(f"  Doppler symbols (M): {system.M}")
    print(f"  T_sym: {system.T_sym * 1e6:.1f} us")

    f_D = system.ue_v * np.cos(system.ue_theta) / lambda_wave
    print(f"  True Doppler freq: {f_D:.1f} Hz")
    print(f"  True velocity component: {f_D * lambda_wave:.1f} m/s")

    # BER test
    print(f"\n--- BER Test (SNR=18dB) ---")
    errs, tx_b, rx_b = system.run_ber_trial(10, 18)
    print(f"  TX bits: {tx_b}")
    print(f"  RX bits: {rx_b}")
    print(f"  Errors: {errs}/{len(tx_b)}")

    # RMSE test
    print(f"\n--- RMSE Test (SNR=18dB, M={system.M}) ---")
    d_hat, theta_hat, v_hat, d_true, theta_true, v_true, tx_b, rx_b = \
        system.run_rmse_trial(10, 18)

    x_est = d_hat * np.cos(theta_hat)
    y_est = d_hat * np.sin(theta_hat)
    x_true = d_true * np.cos(theta_true)
    y_true = d_true * np.sin(theta_true)
    pos_err = np.sqrt((x_est - x_true) ** 2 + (y_est - y_true) ** 2)

    print(f"  d_hat={d_hat:.2f}m (true={d_true:.1f}m)")
    print(f"  theta_hat={theta_hat * 180 / np.pi:.1f}deg (true={theta_true * 180 / np.pi:.1f}deg)")
    print(f"  v_hat={v_hat:.2f} m/s (true v·cos(theta)={v_true * np.cos(theta_true):.2f} m/s)")
    print(f"  Position error: {pos_err:.2f} m")

    # Multiple RMSE trials
    print(f"\n--- Quick RMSE sweep (5 trials, SNR=15-20dB) ---")
    for snr in range(15, 21):
        pos_errs = []
        vel_errs = []
        for _ in range(5):
            i_tx = np.random.randint(0, Q)
            d_h, th_h, v_h, d_t, th_t, v_t, _, _ = system.run_rmse_trial(i_tx, snr)
            xe = d_h * np.cos(th_h); ye = d_h * np.sin(th_h)
            xt = d_t * np.cos(th_t); yt = d_t * np.sin(th_t)
            pos_errs.append(np.sqrt((xe - xt) ** 2 + (ye - yt) ** 2))
            vel_errs.append(abs(v_h - v_t * np.cos(th_t)))
        print(f"  SNR={snr}dB: pos RMSE={np.mean(pos_errs):.2f}m, "
              f"vel err={np.mean(vel_errs):.2f} m/s")
