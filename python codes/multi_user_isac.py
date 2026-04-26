"""
Multi-User (MU) OFDM-ISAC System
=================================
Extends the single-user OFDM-ISAC pipeline to support K simultaneous UEs
using Zadoff-Chu Sequence Division (different root indices per UE).

Architecture:
  - Each UE is assigned a unique ZC root index (e.g., root=1 and root=3).
  - The BS correlates the received superimposed signal against each root
    independently to decode bitstreams and estimate locations.
  - ZC sequences with different roots have near-zero cross-correlation,
    preventing inter-user interference.

Reference: Extension of Paper Sec. II-III for multi-user scenario.
"""

import numpy as np
import time
from system_params import *
from orthogonal_sequences import initialize_sequences_for_root, generate_gray_code
from channel_noise import ChannelModel, NoiseModel, snr_db_to_linear
from ofdm_modem import OFDMTransmitter, OFDMReceiver
from sequence_detection import SequenceDetector
from dbscan_clustering import ClusteringLocalizationEngine


class MultiUserISACSystem:
    """
    Orchestrates K independent OFDMISACSystem instances to simulate
    a multi-user ISAC scenario with superimposed transmissions.
    """

    def __init__(self, BW=30.72e6,
                 ue_distances=None, ue_angles=None, ue_zc_roots=None):
        """
        Args:
            BW: System bandwidth [Hz]
            ue_distances: list of K distances [m]    (defaults to UE_DISTANCES)
            ue_angles:    list of K AoA [rad]        (defaults to UE_ANGLES_RAD)
            ue_zc_roots:  list of K ZC root indices  (defaults to UE_ZC_ROOTS)
        """
        self.BW = BW
        self.Ts = 1 / BW

        # Per-UE parameters
        self.distances = ue_distances if ue_distances is not None else UE_DISTANCES
        self.angles    = ue_angles    if ue_angles    is not None else UE_ANGLES_RAD
        self.zc_roots  = ue_zc_roots  if ue_zc_roots  is not None else UE_ZC_ROOTS
        self.K         = len(self.distances)

        assert len(self.angles)   == self.K, "Angles list must match number of UEs"
        assert len(self.zc_roots) == self.K, "ZC roots list must match number of UEs"

        # Shared OFDM parameters
        self.NFFT = NFFT
        self.NCP  = NCP

        # Build per-UE components
        self.per_ue = []
        for k in range(self.K):
            seq_dict = initialize_sequences_for_root(self.zc_roots[k])
            seq_info = seq_dict['ZC']
            gray_enc, idx_to_bits = generate_gray_code(q_bits)

            ue = {
                'seq_dict':   seq_dict,
                'seq_info':   seq_info,
                'gray_enc':   gray_enc,
                'idx_to_bits': idx_to_bits,
                'Ns':         seq_info['Ns'],
                'NCS':        seq_info['NCS'],
                'tx':         OFDMTransmitter(NFFT=NFFT, NCP=NCP),
                'rx':         OFDMReceiver(NFFT=NFFT, NCP=NCP),
                'channel':    ChannelModel(d0=self.distances[k],
                                           theta=self.angles[k]),
                'detector':   SequenceDetector(J=J),
                'clustering': ClusteringLocalizationEngine(
                    NCS=seq_info['NCS'], J=J, lambda_wave=lambda_wave,
                    r_ant=r_ant, eta_threshold=eta_threshold,
                    epsilon=epsilon, min_samples=min_samples),
                'd0':         self.distances[k],
                'theta':      self.angles[k],
                'zc_root':    self.zc_roots[k],
            }
            self.per_ue.append(ue)

    # ------------------------------------------------------------------
    #  BER trial (multi-user)
    # ------------------------------------------------------------------

    def run_ber_trial(self, i_tx_list, SNR_dB):
        """
        Single multi-user BER trial.

        1. Each UE transmits its own OFDM symbol (with its own ZC root).
        2. All signals pass through independent channels and are
           summed at the BS antenna array (superposition).
        3. The BS demodulates and detects each UE independently by
           correlating against that UE's root sequence.

        Args:
            i_tx_list: list of K sequence indices, one per UE
            SNR_dB: Transmit SNR [dB]

        Returns:
            per_ue_results: list of K dicts with 'bit_errors', 'tx_bits', 'rx_bits'
        """
        SNR_lin   = snr_db_to_linear(SNR_dB)
        Ns0       = self.per_ue[0]['Ns']
        noise_var = NoiseModel.calculate_noise_variance(SNR_lin, self.NFFT,
                                                         self.NCP, Ns0)
        sig_len = self.NFFT + self.NCP

        # -- 1. Generate superimposed received signal ---------------------
        r_total = np.zeros((J, sig_len), dtype=complex)

        for k in range(self.K):
            ue = self.per_ue[k]
            i_tx = i_tx_list[k]

            # Transmit
            s_i = ue['seq_info']['S'][i_tx, :]
            x_CP = ue['tx'].modulate(s_i, ue['Ns'])

            # Channel (per-UE, independent realisation)
            h_delay, tau_int = ue['channel'].generate_channel(self.Ts,
                                                               c_light=c_light)
            L_ch = ue['channel'].L

            # Receive through channel (NO per-UE noise -- add once at end)
            for jj in range(J):
                r_j = np.zeros(sig_len, dtype=complex)
                for l_idx in range(L_ch):
                    tau_l = tau_int[l_idx]
                    if tau_l < sig_len:
                        x_delayed = np.concatenate([np.zeros(tau_l),
                                                    x_CP[:sig_len - tau_l]])
                        r_j += h_delay[jj, l_idx] * x_delayed
                r_total[jj, :] += r_j  # Sum across UEs (superposition)

        # Add AWGN once across the combined signal
        noise = np.sqrt(noise_var / 2) * (
            np.random.randn(J, sig_len) + 1j * np.random.randn(J, sig_len))
        r_total += noise

        # -- 2. Detect each UE independently ------------------------------
        per_ue_results = []
        for k in range(self.K):
            ue = self.per_ue[k]
            i_tx = i_tx_list[k]
            tx_bits = ue['idx_to_bits'][i_tx, :]

            # OFDM demod
            y_all = ue['rx'].demodulate(r_total, ue['Ns'], J=J)

            # Detect against THIS UE's root sequence
            i_hat, kappa_hat = ue['detector'].detect_sequence(
                y_all, ue['seq_info']['s0'], ue['NCS'], Q, use_diversity=True
            )
            _, rx_bits = ue['detector'].demodulate(i_hat, ue['gray_enc'],
                                                    ue['idx_to_bits'])

            bit_errors = int(np.sum(rx_bits != tx_bits))
            per_ue_results.append({
                'bit_errors': bit_errors,
                'tx_bits':    tx_bits.copy(),
                'rx_bits':    rx_bits.copy(),
                'i_tx':       i_tx,
                'i_hat':      int(i_hat),
            })

        return per_ue_results

    # ------------------------------------------------------------------
    #  RMSE trial (multi-user)
    # ------------------------------------------------------------------

    def run_rmse_trial(self, i_tx_list, SNR_dB, n_sequences=5):
        """
        Single multi-user RMSE trial.

        1. For M consecutive time slots, each UE transmits its symbol.
        2. All UE signals are superimposed through independent channels.
        3. The BS demodulates and runs DBSCAN clustering per root to
           estimate each UE's (distance, angle).

        Args:
            i_tx_list: list of K initial sequence indices
            SNR_dB: Transmit SNR [dB]
            n_sequences: Number of symbols per trial (M)

        Returns:
            per_ue_results: list of K dicts with 'd_hat', 'theta_hat',
                           'd_true', 'theta_true', 'tx_bits', 'rx_bits'
        """
        SNR_lin   = snr_db_to_linear(SNR_dB)
        Ns0       = self.per_ue[0]['Ns']
        noise_var = NoiseModel.calculate_noise_variance(SNR_lin, self.NFFT,
                                                         self.NCP, Ns0)
        sig_len = self.NFFT + self.NCP

        # -- 1. Collect M sequential superimposed symbols -----------------
        # We store the demodulated output per-root across all M symbols.
        y_multi_per_ue = {k: np.zeros((n_sequences * J, Ns0), dtype=complex)
                          for k in range(self.K)}

        for m in range(n_sequences):
            # Build superimposed received signal for this time slot
            r_total = np.zeros((J, sig_len), dtype=complex)

            for k in range(self.K):
                ue = self.per_ue[k]
                i_tx_cur = (i_tx_list[k] + m) % Q

                s_i = ue['seq_info']['S'][i_tx_cur, :]
                x_CP = ue['tx'].modulate(s_i, ue['Ns'])

                # Independent channel per UE per time slot (Rayleigh fading)
                h_delay, tau_int = ue['channel'].generate_channel(self.Ts,
                                                                   c_light=c_light)
                L_ch = ue['channel'].L

                for jj in range(J):
                    r_j = np.zeros(sig_len, dtype=complex)
                    for l_idx in range(L_ch):
                        tau_l = tau_int[l_idx]
                        if tau_l < sig_len:
                            x_delayed = np.concatenate(
                                [np.zeros(tau_l), x_CP[:sig_len - tau_l]])
                            r_j += h_delay[jj, l_idx] * x_delayed
                    r_total[jj, :] += r_j

            # AWGN once
            noise = np.sqrt(noise_var / 2) * (
                np.random.randn(J, sig_len) + 1j * np.random.randn(J, sig_len))
            r_total += noise

            # Demod per root and store
            for k in range(self.K):
                ue = self.per_ue[k]
                y_all = ue['rx'].demodulate(r_total, ue['Ns'], J=J)
                y_multi_per_ue[k][m * J:(m + 1) * J, :] = y_all

        # -- 2. Per-UE detection + clustering -----------------------------
        per_ue_results = []
        for k in range(self.K):
            ue = self.per_ue[k]
            i_tx = i_tx_list[k]
            tx_bits = ue['idx_to_bits'][i_tx, :]
            d_true  = float(ue['d0'])
            theta_true = float(ue['theta'])

            y_multi = y_multi_per_ue[k]

            # Detect from first symbol
            y_first = y_multi[0:J, :]
            i_hat, kappa_hat = ue['detector'].detect_sequence(
                y_first, ue['seq_info']['s0'], ue['NCS'], Q, use_diversity=True
            )
            _, rx_bits = ue['detector'].demodulate(i_hat, ue['gray_enc'],
                                                    ue['idx_to_bits'])

            # DBSCAN clustering
            d_hat, theta_hat, _ = ue['clustering'].estimate_localization(
                y_multi, ue['seq_info']['s0'], i_hat, kappa_hat,
                self.NFFT, ue['Ns'], self.Ts, Q,
                num_seq=n_sequences, angle_grid_size=361
            )

            # Fallback
            if d_hat is None or d_hat <= 0:
                D_raw = ue['clustering'].collect_data(
                    y_multi, ue['seq_info']['s0'], i_hat,
                    self.NFFT, ue['Ns'], self.Ts, Q,
                    num_seq=n_sequences, angle_grid_size=361
                )
                best_idx = np.argmax(D_raw[:, 2])
                d_hat     = float(D_raw[best_idx, 0])
                theta_hat = float(D_raw[best_idx, 1])

            per_ue_results.append({
                'd_hat':      d_hat,
                'theta_hat':  theta_hat,
                'd_true':     d_true,
                'theta_true': theta_true,
                'tx_bits':    tx_bits.copy(),
                'rx_bits':    rx_bits.copy(),
                'i_tx':       i_tx,
                'i_hat':      int(i_hat),
                'bit_errors': int(np.sum(rx_bits != tx_bits)),
            })

        return per_ue_results


# ===========================================================================
#  Demonstration / validation functions
# ===========================================================================

def demo_mu_ber(system, num_trials=500, snr_vec=None):
    """Run multi-user BER evaluation."""
    if snr_vec is None:
        snr_vec = list(range(13, 21))

    print("\n" + "=" * 72)
    print("  Multi-User BER Evaluation")
    print("=" * 72)
    for k in range(system.K):
        ue = system.per_ue[k]
        print(f"  UE{k+1}: root={ue['zc_root']}, "
              f"d0={ue['d0']:.1f}m, "
              f"angle={ue['theta']*180/np.pi:.1f}deg")
    print(f"  Trials per SNR: {num_trials}")
    print("-" * 72)

    header = f"{'SNR [dB]':>10}"
    for k in range(system.K):
        header += f" | {'UE'+str(k+1)+' BER':>14}"
    print(header)
    print("-" * 72)

    for snr in snr_vec:
        errors = [0] * system.K
        total_bits = [0] * system.K

        for trial in range(num_trials):
            i_tx_list = [np.random.randint(0, Q) for _ in range(system.K)]
            results = system.run_ber_trial(i_tx_list, snr)

            for k in range(system.K):
                errors[k]     += results[k]['bit_errors']
                total_bits[k] += q_bits

        row = f"{snr:10d}"
        for k in range(system.K):
            ber = errors[k] / total_bits[k] if total_bits[k] > 0 else 0
            row += f" | {ber:14.4e}"
        print(row)

    print("=" * 72)


def demo_mu_rmse(system, num_trials=100, snr_vec=None, n_sequences=5):
    """Run multi-user RMSE evaluation."""
    if snr_vec is None:
        snr_vec = list(range(13, 21))

    print("\n" + "=" * 72)
    print("  Multi-User RMSE Evaluation")
    print("=" * 72)
    for k in range(system.K):
        ue = system.per_ue[k]
        print(f"  UE{k+1}: root={ue['zc_root']}, "
              f"d0={ue['d0']:.1f}m, "
              f"angle={ue['theta']*180/np.pi:.1f}deg")
    print(f"  Trials per SNR: {num_trials}, M={n_sequences} symbols")
    print("-" * 72)

    header = f"{'SNR [dB]':>10}"
    for k in range(system.K):
        header += f" | {'UE'+str(k+1)+' RMSE [m]':>14}"
    print(header)
    print("-" * 72)

    for snr in snr_vec:
        sq_err = [0.0] * system.K
        valid  = [0]   * system.K

        for trial in range(num_trials):
            i_tx_list = [np.random.randint(0, Q) for _ in range(system.K)]
            results = system.run_rmse_trial(i_tx_list, snr,
                                            n_sequences=n_sequences)

            for k in range(system.K):
                r = results[k]
                if r['d_hat'] is not None and np.isfinite(r['d_hat']):
                    x_e = r['d_hat']  * np.cos(r['theta_hat'])
                    y_e = r['d_hat']  * np.sin(r['theta_hat'])
                    x_t = r['d_true'] * np.cos(r['theta_true'])
                    y_t = r['d_true'] * np.sin(r['theta_true'])
                    sq_err[k] += (x_e - x_t)**2 + (y_e - y_t)**2
                    valid[k]  += 1

            if (trial + 1) % 50 == 0:
                status = f"  SNR={snr:2d}dB trial {trial+1}/{num_trials}:"
                for k in range(system.K):
                    rmse_k = np.sqrt(sq_err[k] / valid[k]) if valid[k] > 0 else np.inf
                    status += f"  UE{k+1}={rmse_k:.2f}m"
                print(status)

        row = f"{snr:10d}"
        for k in range(system.K):
            rmse_k = np.sqrt(sq_err[k] / valid[k]) if valid[k] > 0 else np.inf
            row += f" | {rmse_k:14.2f}"
        print(row)

    print("=" * 72)


def demo_cross_correlation():
    """
    Demonstrate that ZC roots 1 and 3 are nearly orthogonal:
    high auto-correlation peak, near-zero cross-correlation.
    """
    print("\n" + "=" * 72)
    print("  ZC Cross-Correlation Analysis")
    print("=" * 72)

    from orthogonal_sequences import generate_zadoff_chu

    roots = UE_ZC_ROOTS
    seqs = {r: generate_zadoff_chu(Ns_ZC, u=r) for r in roots}

    for r1 in roots:
        for r2 in roots:
            xcorr = np.abs(np.dot(seqs[r1].conj(), seqs[r2]))
            norm  = np.sqrt(np.dot(seqs[r1].conj(), seqs[r1]).real *
                            np.dot(seqs[r2].conj(), seqs[r2]).real)
            xcorr_norm = xcorr / norm
            label = "AUTO" if r1 == r2 else "CROSS"
            print(f"  root {r1} x root {r2}:  |corr| = {xcorr_norm:.6f}  [{label}]")

    print("=" * 72)


# ===========================================================================
#  Main
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(42)
    t0 = time.time()

    print("=" * 68)
    print("      Multi-User OFDM-ISAC System -- Validation Run")
    print("=" * 68)

    # 1. Cross-correlation proof
    demo_cross_correlation()

    # 2. Build 2-UE system
    mu_system = MultiUserISACSystem(
        BW=30.72e6,
        ue_distances=UE_DISTANCES,
        ue_angles=UE_ANGLES_RAD,
        ue_zc_roots=UE_ZC_ROOTS,
    )

    # 3. Quick BER check (500 trials)
    demo_mu_ber(mu_system, num_trials=500,
                snr_vec=[13, 15, 17, 19, 20])

    # 4. Quick RMSE check (100 trials)
    demo_mu_rmse(mu_system, num_trials=100,
                 snr_vec=[15, 17, 20], n_sequences=5)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
