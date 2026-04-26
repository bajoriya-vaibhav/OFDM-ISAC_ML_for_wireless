"""
Mobile UE Sensing RMSE Simulation
Simulates position RMSE and velocity RMSE for a single mobile UE.
Comparable to sensing_rmse.py but uses Doppler-aware channel + velocity estimation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from system_params import *
from mobile_isac import MobileOFDMISACSystem


def run_mobile_rmse_simulation(BW, SNR_dB_vec, ue_v=None,
                                num_trials=200, n_sequences=None):
    """
    Run mobile RMSE simulation for a specific bandwidth.

    Returns:
        pos_rmse: Position RMSE per SNR point [m]
        vel_rmse: Velocity RMSE per SNR point [m/s]
    """
    if n_sequences is None:
        n_sequences = M_doppler
    if ue_v is None:
        ue_v = v_ue

    print(f"\nSimulating BW={BW/1e6:.2f} MHz (Mobile, v={ue_v:.0f} m/s) ...")

    system = MobileOFDMISACSystem(seq_type='ZC', BW=BW, ue_v=ue_v)
    v_true_radial = ue_v * np.cos(theta_true)  # Radial velocity component

    pos_rmse = np.zeros(len(SNR_dB_vec))
    vel_rmse = np.zeros(len(SNR_dB_vec))

    for snr_idx, SNR_dB in enumerate(SNR_dB_vec):
        sq_err_pos = 0.0
        sq_err_vel = 0.0
        valid = 0

        for trial in range(num_trials):
            i_tx = np.random.randint(0, Q)
            d_hat, theta_hat, v_hat, d_true, theta_true_v, v_true, tx_b, rx_b = \
                system.run_rmse_trial(i_tx, SNR_dB, n_sequences=n_sequences)

            if d_hat is not None and np.isfinite(d_hat):
                # Position error (Cartesian)
                x_e = d_hat * np.cos(theta_hat)
                y_e = d_hat * np.sin(theta_hat)
                x_t = d_true * np.cos(theta_true_v)
                y_t = d_true * np.sin(theta_true_v)
                sq_err_pos += (x_e - x_t) ** 2 + (y_e - y_t) ** 2

                # Velocity error (radial component)
                sq_err_vel += (v_hat - v_true_radial) ** 2
                valid += 1

            if (trial + 1) % 100 == 0:
                p_rmse = np.sqrt(sq_err_pos / valid) if valid > 0 else np.inf
                v_rmse_cur = np.sqrt(sq_err_vel / valid) if valid > 0 else np.inf
                print(f"  BW={BW/1e6:.2f}MHz, SNR={SNR_dB:2d}dB: "
                      f"Trial {trial+1:5d}/{num_trials}  "
                      f"posRMSE={p_rmse:.2f}m  velRMSE={v_rmse_cur:.2f}m/s")

        pos_rmse[snr_idx] = np.sqrt(sq_err_pos / valid) if valid > 0 else np.inf
        vel_rmse[snr_idx] = np.sqrt(sq_err_vel / valid) if valid > 0 else np.inf

        print(f"  BW={BW/1e6:.2f}MHz, SNR={SNR_dB:2d}dB FINAL: "
              f"posRMSE={pos_rmse[snr_idx]:.2f}m  "
              f"velRMSE={vel_rmse[snr_idx]:.2f}m/s  (valid={valid})")

    return pos_rmse, vel_rmse


def mobile_sensing_rmse():
    """Run mobile sensing RMSE and plot."""
    print("=" * 70)
    print("Mobile UE OFDM-ISAC Sensing RMSE Simulation")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Distance: {d0:.1f} m")
    print(f"  Angle: {theta_true * 180 / np.pi:.1f} deg")
    print(f"  Velocity: {v_ue:.1f} m/s ({v_ue * 3.6:.0f} km/h)")
    print(f"  Doppler symbols (M): {M_doppler}")
    print(f"  Trials per SNR: 500")
    print(f"  SNR range: {SNR_dB_min}-{SNR_dB_max} dB")

    SNR_dB_vec = np.arange(SNR_dB_min, SNR_dB_max + 1, SNR_dB_step)

    # Run for single bandwidth (30.72 MHz) to keep within 1 hr
    BW = 30.72e6
    pos_rmse, vel_rmse = run_mobile_rmse_simulation(
        BW, SNR_dB_vec, num_trials=500, n_sequences=M_doppler)

    # Print results table
    print("\n" + "=" * 60)
    print("MOBILE RMSE RESULTS TABLE (BW=30.72 MHz)")
    print("=" * 60)
    print(f"{'SNR [dB]':>10} | {'Pos RMSE [m]':>15} | {'Vel RMSE [m/s]':>15}")
    print("-" * 60)
    for i, snr in enumerate(SNR_dB_vec):
        print(f"{snr:10.0f} | {pos_rmse[i]:15.2f} | {vel_rmse[i]:15.2f}")
    print("=" * 60)

    # ---- Plot 1: Position RMSE vs SNR ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Position RMSE
    ax1.plot(SNR_dB_vec, pos_rmse, color='#e63946', marker='o',
             linestyle='-', linewidth=2, markersize=8,
             label=f'Mobile (v={v_ue:.0f} m/s)')

    ax1.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    ax1.grid(True, which='minor', linestyle=':', alpha=0.3, color='#e0e0e0')
    ax1.minorticks_on()
    ax1.set_xlabel('Transmit SNR [dB]', fontsize=13)
    ax1.set_ylabel('RMSE Localization Error [m]', fontsize=13)
    ax1.set_xlim([SNR_dB_min, SNR_dB_max])
    ax1.set_ylim(bottom=0)
    ax1.set_title(f'Position RMSE (BW={BW/1e6:.2f} MHz)', fontsize=13)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9,
               shadow=True, fancybox=True)

    # Right: Velocity RMSE
    ax2.plot(SNR_dB_vec, vel_rmse, color='#2a9d8f', marker='D',
             linestyle='-', linewidth=2, markersize=8,
             label=f'Velocity RMSE (v_true={v_ue:.0f} m/s)')

    # Add true velocity reference line
    ax2.axhline(y=v_ue * np.cos(theta_true), color='black',
                linestyle=':', linewidth=1, alpha=0.5,
                label=f'v*cos(theta) = {v_ue*np.cos(theta_true):.1f} m/s')

    ax2.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    ax2.grid(True, which='minor', linestyle=':', alpha=0.3, color='#e0e0e0')
    ax2.minorticks_on()
    ax2.set_xlabel('Transmit SNR [dB]', fontsize=13)
    ax2.set_ylabel('RMSE Velocity Error [m/s]', fontsize=13)
    ax2.set_xlim([SNR_dB_min, SNR_dB_max])
    ax2.set_ylim(bottom=0)
    ax2.set_title(f'Velocity RMSE (BW={BW/1e6:.2f} MHz)', fontsize=13)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9,
               shadow=True, fancybox=True)

    plt.suptitle(f'Mobile UE Sensing Performance (d={d0:.0f}m, v={v_ue:.0f}m/s)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig('./mobile_fig_rmse_sensing.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to: ./mobile_fig_rmse_sensing.png")
    plt.show()

    return SNR_dB_vec, pos_rmse, vel_rmse


if __name__ == "__main__":
    SNR_vec, pos_rmse, vel_rmse = mobile_sensing_rmse()

    np.savez('./mobile_rmse_results.npz',
             SNR=SNR_vec, pos_rmse=pos_rmse, vel_rmse=vel_rmse)
    print("\nResults saved to: ./mobile_rmse_results.npz")
