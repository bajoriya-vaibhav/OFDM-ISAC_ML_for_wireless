"""
Multi-User Communication BER Simulation
Simulates BER performance for K UEs with superimposed ZC-root transmissions.
Includes combined RMS-average BER curve showing overall system performance.
Mirrors communication_ber.py for the multi-user case.
"""

import numpy as np
import matplotlib.pyplot as plt
from system_params import *
from multi_user_isac import MultiUserISACSystem


def run_mu_ber_simulation(mu_system, SNR_dB_vec, num_trials=5000):
    """
    Run multi-user BER simulation.

    Args:
        mu_system: MultiUserISACSystem instance
        SNR_dB_vec: Array of SNR values in dB
        num_trials: Monte Carlo trials per SNR point

    Returns:
        BER_per_ue: dict  {ue_index: np.array of BER per SNR}
    """
    K = mu_system.K
    BER_per_ue = {k: np.zeros(len(SNR_dB_vec)) for k in range(K)}

    for snr_idx, SNR_dB in enumerate(SNR_dB_vec):
        errors = [0] * K
        total_bits = [0] * K

        for trial in range(num_trials):
            i_tx_list = [np.random.randint(0, Q) for _ in range(K)]
            results = mu_system.run_ber_trial(i_tx_list, SNR_dB)

            for k in range(K):
                errors[k] += results[k]['bit_errors']
                total_bits[k] += q_bits

            if (trial + 1) % 500 == 0:
                msg = f"  SNR={SNR_dB:2d}dB: Trial {trial+1:5d}/{num_trials}"
                for k in range(K):
                    ber_k = errors[k] / total_bits[k]
                    msg += f"  UE{k+1}={ber_k:.4e}"
                print(msg)

        for k in range(K):
            BER_per_ue[k][snr_idx] = errors[k] / total_bits[k]

        row = f"  SNR={SNR_dB:2d}dB FINAL:"
        for k in range(K):
            row += f"  UE{k+1} BER={BER_per_ue[k][snr_idx]:.4e}"
        print(row)

    return BER_per_ue


def compute_rms_average(per_ue_dict, num_ues):
    """Compute RMS average across all UEs: sqrt(mean(val_k^2))."""
    n_points = len(per_ue_dict[0])
    rms = np.zeros(n_points)
    for i in range(n_points):
        sq_sum = sum(per_ue_dict[k][i] ** 2 for k in range(num_ues))
        rms[i] = np.sqrt(sq_sum / num_ues)
    return rms


def mu_communication_ber():
    """Run complete multi-user BER simulation and plot."""
    print("=" * 70)
    print("Multi-User OFDM-ISAC Communication BER Simulation")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  System bandwidth: 30.72 MHz")
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
    print(f"  Monte Carlo trials per point: {num_trials_BER}")

    # SNR range
    SNR_dB_vec = np.arange(SNR_dB_min, SNR_dB_max + 1, SNR_dB_step)
    print(f"  SNR range: {SNR_dB_min}-{SNR_dB_max} dB (step: {SNR_dB_step} dB)")

    # Build system
    mu_system = MultiUserISACSystem(
        BW=30.72e6,
        ue_distances=UE_DISTANCES,
        ue_angles=UE_ANGLES_RAD,
        ue_zc_roots=UE_ZC_ROOTS,
    )

    # Simulate
    BER_per_ue = run_mu_ber_simulation(mu_system, SNR_dB_vec,
                                        num_trials=5000)

    # Combined RMS average
    BER_rms = compute_rms_average(BER_per_ue, mu_system.K)

    # Print results table
    print("\n" + "=" * 70)
    print("MU-BER RESULTS TABLE")
    print("=" * 70)
    header = f"{'SNR [dB]':>10}"
    for k in range(mu_system.K):
        ue = mu_system.per_ue[k]
        header += f" | {'UE'+str(k+1)+' (r='+str(ue['zc_root'])+')':>14}"
    header += f" | {'RMS Avg':>14}"
    print(header)
    print("-" * 70)
    for i, snr in enumerate(SNR_dB_vec):
        row = f"{snr:10.0f}"
        for k in range(mu_system.K):
            row += f" | {BER_per_ue[k][i]:14.4e}"
        row += f" | {BER_rms[i]:14.4e}"
        print(row)
    print("=" * 70)

    # ---- Plot ----
    plt.figure(figsize=(10, 7))

    colors = ['#1d3557', '#e63946', '#2a9d8f', '#f4a261']
    markers = ['s', 'o', 'D', '^']
    linestyles = ['-', '--', '-.', ':']

    for k in range(mu_system.K):
        ue = mu_system.per_ue[k]
        label = (f"UE{k+1} (root={ue['zc_root']}, "
                 f"d={ue['d0']:.0f}m, "
                 f"{ue['theta']*180/np.pi:.0f}deg)")
        plt.semilogy(SNR_dB_vec, BER_per_ue[k],
                      color=colors[k % len(colors)],
                      marker=markers[k % len(markers)],
                      linestyle=linestyles[k % len(linestyles)],
                      linewidth=2, markersize=8, label=label)

    # Combined RMS average curve
    plt.semilogy(SNR_dB_vec, BER_rms, color='black',
                  marker='*', linestyle='-', linewidth=2.5,
                  markersize=10, label='Combined RMS Avg', zorder=5)

    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Transmit SNR [dB]', fontsize=13)
    plt.ylabel('BER', fontsize=13)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9,
               shadow=True, fancybox=True)
    plt.xlim([SNR_dB_min, SNR_dB_max])
    plt.ylim([1e-5, 1e0])
    plt.title(f'Multi-User BER ({NUM_UES} UEs): OFDM-ISAC with ZC Sequence Division',
              fontsize=13)
    plt.tight_layout()

    plt.savefig('./mu_fig_ber_communication.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to: ./mu_fig_ber_communication.png")

    plt.show()

    return SNR_dB_vec, BER_per_ue, BER_rms


if __name__ == "__main__":
    SNR_vec, BER_per_ue, BER_rms = mu_communication_ber()

    # Save results
    save_dict = {'SNR': SNR_vec, 'BER_RMS': BER_rms}
    for k in range(NUM_UES):
        save_dict[f'BER_UE{k+1}'] = BER_per_ue[k]
    np.savez('./mu_ber_results.npz', **save_dict)
    print("\nResults saved to: ./mu_ber_results.npz")
