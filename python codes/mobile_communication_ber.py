"""
Mobile UE Communication BER Simulation
Simulates BER for a single mobile UE with Doppler channel.
Comparable to communication_ber.py but uses Doppler-aware channel.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from system_params import *
from mobile_isac import MobileOFDMISACSystem


def run_mobile_ber_simulation(system, SNR_dB_vec, num_trials=5000):
    """Run BER simulation for mobile UE."""
    BER_vec = np.zeros(len(SNR_dB_vec))

    for snr_idx, SNR_dB in enumerate(SNR_dB_vec):
        errors = 0
        total_bits = 0

        for trial in range(num_trials):
            i_tx = np.random.randint(0, Q)
            errs, _, _ = system.run_ber_trial(i_tx, SNR_dB)
            errors += errs
            total_bits += q_bits

            if (trial + 1) % 500 == 0:
                ber = errors / total_bits
                print(f"  SNR={SNR_dB:2d}dB: Trial {trial+1:5d}/{num_trials}  "
                      f"BER={ber:.4e}")

        BER_vec[snr_idx] = errors / total_bits
        print(f"  SNR={SNR_dB:2d}dB FINAL: BER={BER_vec[snr_idx]:.4e}")

    return BER_vec


def mobile_communication_ber():
    """Run complete mobile BER simulation, plot, and compare with static."""
    print("=" * 70)
    print("Mobile UE OFDM-ISAC Communication BER Simulation")
    print("=" * 70)

    # Build system
    system = MobileOFDMISACSystem(seq_type='ZC', BW=30.72e6)

    print(f"Parameters:")
    print(f"  Distance: {system.ue_d0:.1f} m")
    print(f"  Angle: {system.ue_theta * 180 / np.pi:.1f} deg")
    print(f"  Velocity: {system.ue_v:.1f} m/s ({system.ue_v * 3.6:.0f} km/h)")
    print(f"  Bandwidth: 30.72 MHz")
    print(f"  Trials per SNR: 5000")

    SNR_dB_vec = np.arange(SNR_dB_min, SNR_dB_max + 1, SNR_dB_step)
    print(f"  SNR range: {SNR_dB_min}-{SNR_dB_max} dB")

    # Run
    BER_mobile = run_mobile_ber_simulation(system, SNR_dB_vec, num_trials=5000)

    # Also run static for comparison
    print("\n--- Static comparison ---")
    from ofdm_isac import OFDMISACSystem
    static_sys = OFDMISACSystem(seq_type='ZC', BW=30.72e6)
    BER_static = np.zeros(len(SNR_dB_vec))

    for snr_idx, SNR_dB in enumerate(SNR_dB_vec):
        errors = 0
        total_bits = 0
        for trial in range(5000):
            i_tx = np.random.randint(0, Q)
            errs, _, _ = static_sys.run_ber_trial(i_tx, SNR_dB)
            errors += errs
            total_bits += q_bits
        BER_static[snr_idx] = errors / total_bits
        print(f"  Static SNR={SNR_dB:2d}dB: BER={BER_static[snr_idx]:.4e}")

    # Print results table
    print("\n" + "=" * 50)
    print("BER RESULTS TABLE")
    print("=" * 50)
    print(f"{'SNR [dB]':>10} | {'Mobile':>14} | {'Static':>14}")
    print("-" * 50)
    for i, snr in enumerate(SNR_dB_vec):
        print(f"{snr:10.0f} | {BER_mobile[i]:14.4e} | {BER_static[i]:14.4e}")
    print("=" * 50)

    # Plot
    plt.figure(figsize=(10, 7))

    plt.semilogy(SNR_dB_vec, BER_mobile, color='#e63946',
                  marker='o', linestyle='-', linewidth=2, markersize=8,
                  label=f'Mobile (v={system.ue_v:.0f} m/s)')
    plt.semilogy(SNR_dB_vec, BER_static, color='#1d3557',
                  marker='s', linestyle='--', linewidth=2, markersize=8,
                  label='Static (v=0)')

    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Transmit SNR [dB]', fontsize=13)
    plt.ylabel('BER', fontsize=13)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9,
               shadow=True, fancybox=True)
    plt.xlim([SNR_dB_min, SNR_dB_max])
    plt.ylim([1e-5, 1e0])
    plt.title(f'Mobile vs Static BER: OFDM-ISAC (d={system.ue_d0:.0f}m, '
              f'v={system.ue_v:.0f}m/s)', fontsize=13)
    plt.tight_layout()

    plt.savefig('./mobile_fig_ber.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to: ./mobile_fig_ber.png")
    plt.show()

    return SNR_dB_vec, BER_mobile, BER_static


if __name__ == "__main__":
    SNR_vec, BER_mob, BER_stat = mobile_communication_ber()

    np.savez('./mobile_ber_results.npz',
             SNR=SNR_vec, BER_mobile=BER_mob, BER_static=BER_stat)
    print("\nResults saved to: ./mobile_ber_results.npz")
