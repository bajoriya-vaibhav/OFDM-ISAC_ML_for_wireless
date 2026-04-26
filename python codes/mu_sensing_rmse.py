"""
Multi-User Sensing RMSE Simulation
Simulates RMSE localization performance for K UEs across different bandwidths.
Includes combined RMS-average RMSE curve showing overall system performance.
Mirrors sensing_rmse.py for the multi-user case.
"""

import numpy as np
import matplotlib.pyplot as plt
from system_params import *
from multi_user_isac import MultiUserISACSystem


def run_mu_rmse_simulation(BW, SNR_dB_vec, num_trials=500, n_sequences=5):
    """
    Run multi-user RMSE simulation for a specific bandwidth.

    Args:
        BW: System bandwidth [Hz]
        SNR_dB_vec: Array of SNR values in dB
        num_trials: Monte Carlo trials per SNR point
        n_sequences: Symbols per trial (M)

    Returns:
        RMSE_per_ue: dict {ue_index: np.array of RMSE per SNR}
    """
    print(f"\nSimulating BW={BW/1e6:.2f} MHz ...")

    mu_system = MultiUserISACSystem(
        BW=BW,
        ue_distances=UE_DISTANCES,
        ue_angles=UE_ANGLES_RAD,
        ue_zc_roots=UE_ZC_ROOTS,
    )
    K = mu_system.K
    RMSE_per_ue = {k: np.zeros(len(SNR_dB_vec)) for k in range(K)}

    for snr_idx, SNR_dB in enumerate(SNR_dB_vec):
        sq_err = [0.0] * K
        valid = [0] * K

        for trial in range(num_trials):
            i_tx_list = [np.random.randint(0, Q) for _ in range(K)]
            results = mu_system.run_rmse_trial(i_tx_list, SNR_dB,
                                                n_sequences=n_sequences)

            for k in range(K):
                r = results[k]
                if r['d_hat'] is not None and np.isfinite(r['d_hat']):
                    x_e = r['d_hat']  * np.cos(r['theta_hat'])
                    y_e = r['d_hat']  * np.sin(r['theta_hat'])
                    x_t = r['d_true'] * np.cos(r['theta_true'])
                    y_t = r['d_true'] * np.sin(r['theta_true'])
                    sq_err[k] += (x_e - x_t)**2 + (y_e - y_t)**2
                    valid[k] += 1

            if (trial + 1) % 100 == 0:
                msg = (f"  BW={BW/1e6:.2f}MHz, SNR={SNR_dB:2d}dB: "
                       f"Trial {trial+1:5d}/{num_trials}")
                for k in range(K):
                    rmse_k = np.sqrt(sq_err[k]/valid[k]) if valid[k] > 0 else np.inf
                    msg += f"  UE{k+1}={rmse_k:.2f}m"
                print(msg)

        for k in range(K):
            RMSE_per_ue[k][snr_idx] = (np.sqrt(sq_err[k] / valid[k])
                                        if valid[k] > 0 else np.inf)

        row = f"  BW={BW/1e6:.2f}MHz, SNR={SNR_dB:2d}dB FINAL:"
        for k in range(K):
            row += f"  UE{k+1} RMSE={RMSE_per_ue[k][snr_idx]:.2f}m"
        print(row)

    return RMSE_per_ue


def compute_rms_average(per_ue_dict, num_ues):
    """Compute RMS average across all UEs: sqrt(mean(val_k^2))."""
    n_points = len(per_ue_dict[0])
    rms = np.zeros(n_points)
    for i in range(n_points):
        sq_sum = sum(per_ue_dict[k][i] ** 2 for k in range(num_ues))
        rms[i] = np.sqrt(sq_sum / num_ues)
    return rms


def mu_sensing_rmse():
    """Run complete multi-user RMSE simulation and plot."""
    print("=" * 70)
    print("Multi-User OFDM-ISAC Sensing RMSE Simulation")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Sequence type: Zadoff-Chu")
    print(f"  FFT size: {NFFT}")
    print(f"  CP length: {NCP}")
    print(f"  Number of antennas: {J}")
    print(f"  Multipath taps: {L} (local={L_local}, interf={L_interf})")
    print(f"  Index modulation: {q_bits} bits (Q={Q})")
    print(f"  Number of UEs: {NUM_UES}")
    for k in range(NUM_UES):
        print(f"    UE{k+1}: d0={UE_DISTANCES[k]:.1f}m, "
              f"angle={UE_ANGLES_DEG[k]:.1f}deg, "
              f"ZC root={UE_ZC_ROOTS[k]}")
    print(f"  Monte Carlo trials per point: {num_trials_RMSE}")

    # SNR range
    SNR_dB_vec = np.arange(SNR_dB_min, SNR_dB_max + 1, SNR_dB_step)
    print(f"  SNR range: {SNR_dB_min}-{SNR_dB_max} dB (step: {SNR_dB_step} dB)")

    BW_options = bandwidth_options
    print(f"  Bandwidth options: {[f'{bw/1e6:.2f}' for bw in BW_options]} MHz")

    # Simulate for each bandwidth
    RMSE_results = {}
    for BW in BW_options:
        RMSE_results[BW] = run_mu_rmse_simulation(
            BW, SNR_dB_vec, num_trials=500, n_sequences=5)

    # Compute RMS averages per bandwidth
    RMSE_rms = {}
    for BW in BW_options:
        RMSE_rms[BW] = compute_rms_average(RMSE_results[BW], NUM_UES)

    # Print results table -- one table per UE + combined
    for k in range(NUM_UES):
        print(f"\n{'=' * 70}")
        print(f"RMSE RESULTS -- UE{k+1} "
              f"(root={UE_ZC_ROOTS[k]}, d0={UE_DISTANCES[k]:.0f}m, "
              f"angle={UE_ANGLES_DEG[k]:.0f}deg)")
        print("=" * 70)
        print(f"{'SNR [dB]':>10} | {'30.72 MHz':>15} | "
              f"{'61.44 MHz':>15} | {'184.32 MHz':>15}")
        print("-" * 70)
        for i, snr in enumerate(SNR_dB_vec):
            vals = [RMSE_results[bw][k][i] for bw in BW_options]
            print(f"{snr:10.0f} | {vals[0]:15.2f} | {vals[1]:15.2f} | {vals[2]:15.2f}")
        print("=" * 70)

    # Print combined RMS table
    print(f"\n{'=' * 70}")
    print(f"RMSE RESULTS -- Combined RMS Average ({NUM_UES} UEs)")
    print("=" * 70)
    print(f"{'SNR [dB]':>10} | {'30.72 MHz':>15} | "
          f"{'61.44 MHz':>15} | {'184.32 MHz':>15}")
    print("-" * 70)
    for i, snr in enumerate(SNR_dB_vec):
        vals = [RMSE_rms[bw][i] for bw in BW_options]
        print(f"{snr:10.0f} | {vals[0]:15.2f} | {vals[1]:15.2f} | {vals[2]:15.2f}")
    print("=" * 70)

    # ---- Plot 1: Per-UE comparison at BW=30.72MHz + RMS avg ----
    plt.figure(figsize=(10, 7))
    ue_colors = ['#1d3557', '#e63946', '#2a9d8f', '#f4a261']
    ue_markers = ['s', 'o', 'D', '^']
    ue_styles = ['-', '--', '-.', ':']

    BW_ref = BW_options[0]
    for k in range(NUM_UES):
        label = (f"UE{k+1} (root={UE_ZC_ROOTS[k]}, "
                 f"d={UE_DISTANCES[k]:.0f}m, "
                 f"{UE_ANGLES_DEG[k]:.0f}deg)")
        plt.plot(SNR_dB_vec, RMSE_results[BW_ref][k],
                 color=ue_colors[k % len(ue_colors)],
                 marker=ue_markers[k % len(ue_markers)],
                 linestyle=ue_styles[k % len(ue_styles)],
                 linewidth=2, markersize=8, label=label)

    # Combined RMS average curve
    plt.plot(SNR_dB_vec, RMSE_rms[BW_ref], color='black',
             marker='*', linestyle='-', linewidth=2.5,
             markersize=10, label='Combined RMS Avg', zorder=5)

    plt.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    plt.grid(True, which='minor', linestyle=':', alpha=0.3, color='#e0e0e0')
    plt.minorticks_on()
    plt.xlabel('Transmit SNR [dB]', fontsize=13, fontweight='medium')
    plt.ylabel('RMSE Localization Error [m]', fontsize=13, fontweight='medium')
    plt.xlim([SNR_dB_min, SNR_dB_max])
    plt.ylim(bottom=0)
    plt.title(f'Multi-User RMSE Comparison ({NUM_UES} UEs, BW={BW_ref/1e6:.2f} MHz)',
              fontsize=13)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9,
               shadow=True, fancybox=True)
    plt.tight_layout()

    plt.savefig('./mu_fig_rmse_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to: ./mu_fig_rmse_comparison.png")
    plt.show()



    return SNR_dB_vec, RMSE_results, RMSE_rms


if __name__ == "__main__":
    SNR_vec, RMSE_results, RMSE_rms = mu_sensing_rmse()

    # Save results
    save_dict = {'SNR': SNR_vec}
    for BW in bandwidth_options:
        for k in range(NUM_UES):
            key = f'RMSE_UE{k+1}_BW{BW/1e6:.0f}'
            save_dict[key] = RMSE_results[BW][k]
        save_dict[f'RMSE_RMS_BW{BW/1e6:.0f}'] = RMSE_rms[BW]
    np.savez('./mu_rmse_results.npz', **save_dict)
    print("\nResults saved to: ./mu_rmse_results.npz")
