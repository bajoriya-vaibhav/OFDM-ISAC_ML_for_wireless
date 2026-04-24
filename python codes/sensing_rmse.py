"""
Sensing RMSE Simulation
Simulates RMSE performance for localization across different bandwidths
Reference: Paper Fig. 5 and MATLAB simulation code
"""

import numpy as np
import matplotlib.pyplot as plt
from system_params import *
from ofdm_isac import OFDMISACSystem


def calculate_rmse(d_est, theta_est, d_true, theta_true):
    """
    Calculate RMSE for localization
    
    Eq. (18): RMSE = sqrt((d_hat*cos(theta_hat) - d*cos(theta))^2 + 
                          (d_hat*sin(theta_hat) - d*sin(theta))^2)
    
    Args:
        d_est: Estimated distance
        theta_est: Estimated angle
        d_true: True distance
        theta_true: True angle
    
    Returns:
        rmse: RMSE value
    """
    if d_est is None or theta_est is None:
        return np.inf
    
    # Convert to Cartesian coordinates
    x_est = d_est * np.cos(theta_est)
    y_est = d_est * np.sin(theta_est)
    
    x_true = d_true * np.cos(theta_true)
    y_true = d_true * np.sin(theta_true)
    
    # RMSE
    rmse = np.sqrt((x_est - x_true)**2 + (y_est - y_true)**2)
    
    return rmse


def run_rmse_simulation(BW, SNR_dB_vec, seq_type='ZC', num_trials=10000, 
                        n_sequences=5):
    """
    Run RMSE simulation for a specific bandwidth
    
    Args:
        BW: System bandwidth [Hz]
        SNR_dB_vec: Array of SNR values in dB
        seq_type: 'ZC', 'mseq', or 'gold'
        num_trials: Number of Monte Carlo trials per SNR point
        n_sequences: Number of sequence repetitions for clustering
    
    Returns:
        RMSE_vec: RMSE for each SNR point
    """
    print(f"\nSimulating BW={BW/1e6:.2f} MHz...")
    
    # Create system with specific bandwidth
    system = OFDMISACSystem(seq_type=seq_type, BW=BW)
    
    RMSE_vec = np.zeros(len(SNR_dB_vec))
    
    for snr_idx, SNR_dB in enumerate(SNR_dB_vec):
        sq_err_sum   = 0.0
        valid_trials = 0
        
        for trial in range(num_trials):
            # Random sequence index
            i_tx = np.random.randint(0, Q)
            
            # Run trial
            d_hat, theta_hat, d_true, theta_true, _, _ = \
                system.run_rmse_trial(i_tx, SNR_dB, n_sequences=n_sequences)
            
            # Eq (18): squared Euclidean distance in Cartesian
            if d_hat is not None and theta_hat is not None and np.isfinite(d_hat):
                x_est = d_hat  * np.cos(theta_hat)
                y_est = d_hat  * np.sin(theta_hat)
                x_true = d_true * np.cos(theta_true)
                y_true = d_true * np.sin(theta_true)
                sq_err_sum   += (x_est - x_true)**2 + (y_est - y_true)**2
                valid_trials += 1
            
            # Progress — show running RMSE
            if (trial + 1) % 1000 == 0:
                avg_rmse = np.sqrt(sq_err_sum / valid_trials) if valid_trials > 0 else np.inf
                print(f"  BW={BW/1e6:.2f}MHz, SNR={SNR_dB:2d}dB: Trial {trial+1:5d}/{num_trials} "
                      f"- RMSE={avg_rmse:.2f}m (valid={valid_trials})")
        
        RMSE_vec[snr_idx] = np.sqrt(sq_err_sum / valid_trials) if valid_trials > 0 else np.inf
        
        print(f"  BW={BW/1e6:.2f}MHz, SNR={SNR_dB:2d}dB: Final RMSE={RMSE_vec[snr_idx]:.2f}m "
              f"(valid trials={valid_trials}/{num_trials})")
    
    return RMSE_vec


def sensing_rmse():
    """
    Run complete sensing RMSE simulation
    """
    print("="*70)
    print("OFDM-ISAC Sensing RMSE Simulation")
    print("="*70)
    print(f"Parameters:")
    print(f"  Sequence type: Zadoff-Chu")
    print(f"  FFT size: {NFFT}")
    print(f"  CP length: {NCP}")
    print(f"  Number of antennas: {J}")
    print(f"  Total multipath taps: {L}")
    print(f"  Local scattered taps: {L_local}")
    print(f"  Interference taps: {L_interf}")
    print(f"  Index modulation: {q_bits} bits (Q={Q})")
    print(f"  Monte Carlo trials per point: {num_trials_RMSE}")
    
    # SNR range
    SNR_dB_vec = np.arange(SNR_dB_min, SNR_dB_max + 1, SNR_dB_step)
    print(f"  SNR range: {SNR_dB_min}-{SNR_dB_max} dB (step: {SNR_dB_step} dB)")
    
    # Bandwidth options
    BW_options = bandwidth_options
    print(f"  Bandwidth options: {[f'{bw/1e6:.2f}' for bw in BW_options]} MHz")
    
    # Simulate for each bandwidth
    RMSE_results = {}
    for BW in BW_options:
        RMSE_vec = run_rmse_simulation(BW, SNR_dB_vec, seq_type='ZC', 
                                       num_trials=num_trials_RMSE, n_sequences=5)
        RMSE_results[BW] = RMSE_vec
    
    # Print results table
    print("\n" + "="*70)
    print("RMSE RESULTS TABLE")
    print("="*70)
    print(f"{'SNR [dB]':>10} | {'30.72 MHz':>15} | {'61.44 MHz':>15} | {'184.32 MHz':>15}")
    print("-"*70)
    for i, snr in enumerate(SNR_dB_vec):
        rmse_30 = RMSE_results[BW_options[0]][i]
        rmse_61 = RMSE_results[BW_options[1]][i]
        rmse_184 = RMSE_results[BW_options[2]][i]
        print(f"{snr:10.0f} | {rmse_30:15.2f} | {rmse_61:15.2f} | {rmse_184:15.2f}")
    print("="*70)
    
    # Plot
    plt.figure(figsize=(9, 6))
    
    colors = ['#e63946', '#2a9d8f', '#1d3557']
    markers = ['s', 'D', '^']
    line_styles = ['-', '--', '-.']
    labels = ['30.72 MHz', '61.44 MHz', '184.32 MHz']
    
    for i, BW in enumerate(BW_options):
        plt.plot(SNR_dB_vec, RMSE_results[BW], color=colors[i], marker=markers[i],
                 linestyle=line_styles[i], linewidth=2, markersize=7, 
                 label=f'Bandwidth: {labels[i]}')
    
    plt.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    plt.grid(True, which='minor', linestyle=':', alpha=0.3, color='#e0e0e0')
    plt.minorticks_on()
    
    plt.xlabel('Transmit SNR [dB]', fontsize=13, fontweight='medium')
    plt.ylabel('RMSE Localization Error [m]', fontsize=13, fontweight='medium')
    
    plt.xlim([SNR_dB_min, SNR_dB_max])
    plt.ylim([0, 70])
    plt.yticks(np.arange(0, 71, 5))
    
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('./fig5_rmse_sensing.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to: ./fig5_rmse_sensing.png")
    
    plt.show()
    
    return SNR_dB_vec, RMSE_results


if __name__ == "__main__":
    # Run simulation
    SNR_vec, RMSE_results = sensing_rmse()
    
    # Save results
    np.savez('./rmse_results.npz',
             SNR=SNR_vec,
             RMSE_30_72=RMSE_results[30.72e6],
             RMSE_61_44=RMSE_results[61.44e6],
             RMSE_184_32=RMSE_results[184.32e6])
    print("\nResults saved to: ./rmse_results.npz")
