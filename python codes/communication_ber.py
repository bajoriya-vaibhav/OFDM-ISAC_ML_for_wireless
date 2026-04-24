"""
Communication BER Simulation
Simulates BER performance for all three sequence types
Reference: Paper Fig. 4 and MATLAB simulation code
"""

import numpy as np
import matplotlib.pyplot as plt
from system_params import *
from ofdm_isac import OFDMISACSystem


def run_ber_simulation(seq_type, SNR_dB_vec, num_trials=10000, BW=30.72e6):
    """
    Run BER simulation for a specific sequence type
    
    Args:
        seq_type: 'ZC', 'mseq', or 'gold'
        SNR_dB_vec: Array of SNR values in dB
        num_trials: Number of Monte Carlo trials per SNR point
        BW: System bandwidth
    
    Returns:
        BER_vec: BER for each SNR point
    """
    print(f"\nSimulating {seq_type} sequence...")
    
    # Create system
    system = OFDMISACSystem(seq_type=seq_type, BW=BW)
    
    BER_vec = np.zeros(len(SNR_dB_vec))
    
    for snr_idx, SNR_dB in enumerate(SNR_dB_vec):
        total_errors = 0
        total_bits = 0
        
        for trial in range(num_trials):
            # Random sequence index
            i_tx = np.random.randint(0, Q)
            
            # Run trial
            num_bit_errors, _, _ = system.run_ber_trial(i_tx, SNR_dB)
            
            total_errors += num_bit_errors
            total_bits += q_bits
            
            # Progress
            if (trial + 1) % 1000 == 0:
                current_ber = total_errors / total_bits
                print(f"  [{seq_type}] SNR={SNR_dB:2d}dB: Trial {trial+1:5d}/{num_trials} "
                      f"- BER={current_ber:.4e}")
        
        BER_vec[snr_idx] = total_errors / total_bits
        print(f"  [{seq_type}] SNR={SNR_dB:2d}dB: Final BER={BER_vec[snr_idx]:.4e}")
    
    return BER_vec


def communication_ber():
    """
    Run complete communication BER simulation
    Reproduces Fig. 4 from paper
    """
    print("="*70)
    print("OFDM-ISAC Communication BER Simulation")
    print("="*70)
    print(f"Parameters:")
    print(f"  System bandwidth: 30.72 MHz")
    print(f"  FFT size: {NFFT}")
    print(f"  CP length: {NCP}")
    print(f"  Number of antennas: {J}")
    print(f"  Total multipath taps: {L}")
    print(f"  Local scattered taps: {L_local}")
    print(f"  Interference taps: {L_interf}")
    print(f"  Index modulation: {q_bits} bits (Q={Q})")
    print(f"  Monte Carlo trials per point: {num_trials_BER}")
    
    # SNR range
    SNR_dB_vec = np.arange(SNR_dB_min, SNR_dB_max + 1, SNR_dB_step)
    print(f"  SNR range: {SNR_dB_min}-{SNR_dB_max} dB (step: {SNR_dB_step} dB)")
    
    # Simulate all sequences
    BER_ZC = run_ber_simulation('ZC', SNR_dB_vec, num_trials=num_trials_BER)
    BER_gold = run_ber_simulation('gold', SNR_dB_vec, num_trials=num_trials_BER)
    BER_mseq = run_ber_simulation('mseq', SNR_dB_vec, num_trials=num_trials_BER)
    
    # Print results table
    print("\n" + "="*70)
    print("BER RESULTS TABLE")
    print("="*70)
    print(f"{'SNR [dB]':>10} | {'Zadoff-Chu':>12} | {'Gold':>12} | {'m-sequence':>12}")
    print("-"*70)
    for i, snr in enumerate(SNR_dB_vec):
        print(f"{snr:10.0f} | {BER_ZC[i]:12.4e} | {BER_gold[i]:12.4e} | {BER_mseq[i]:12.4e}")
    print("="*70)
    
    # Plot
    plt.figure(figsize=(10, 7))
    plt.semilogy(SNR_dB_vec, BER_ZC, 'k-s', linewidth=2, markersize=8, label='Zadoff-Chu sequence')
    plt.semilogy(SNR_dB_vec, BER_gold, 'b-o', linewidth=2, markersize=8, label='Gold sequence')
    plt.semilogy(SNR_dB_vec, BER_mseq, 'r-^', linewidth=2, markersize=8, label='m-sequence')
    
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Transmit SNR [dB]', fontsize=13)
    plt.ylabel('BER', fontsize=13)
    plt.legend(loc='lower left', fontsize=11)
    plt.xlim([SNR_dB_min, SNR_dB_max])
    plt.ylim([1e-5, 1e0])
    plt.title('Fig. 4: BER Performance of OFDM-ISAC System', fontsize=13)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('./fig4_ber_communication.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to: ./fig4_ber_communication.png")
    
    plt.show()
    
    return SNR_dB_vec, BER_ZC, BER_gold, BER_mseq


if __name__ == "__main__":
    # Run simulation
    SNR_vec, BER_ZC, BER_gold, BER_mseq = communication_ber()
    
    # Save results
    np.savez('./ber_results.npz', 
             SNR=SNR_vec, 
             BER_ZC=BER_ZC, 
             BER_gold=BER_gold, 
             BER_mseq=BER_mseq)
    print("\nResults saved to: ./ber_results.npz")
