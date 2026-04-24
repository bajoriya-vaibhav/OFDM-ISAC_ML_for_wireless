"""
Medium length test for BER and RMSE to verify trends before running full simulations.
Runs 1,000 trials for BER and 100 trials for RMSE.
"""

import numpy as np
import time
from ofdm_isac import OFDMISACSystem
from system_params import *

def test_ber(system, num_trials=1000):
    print("\n--- Testing BER (Zadoff-Chu, 30.72MHz) ---")
    print(f"Running {num_trials} trials per SNR...")
    
    snr_vec = list(range(13, 21))
    ber_results = []
    
    for snr in snr_vec:
        errors = 0
        total_bits = 0
        
        for trial in range(num_trials):
            i_tx = np.random.randint(0, Q)
            num_err, tx_bits, rx_bits = system.run_ber_trial(i_tx, snr)
            errors += num_err
            total_bits += len(tx_bits)
            
        ber = errors / total_bits
        ber_results.append(ber)
        print(f"SNR = {snr:2d} dB | BER = {ber:.4e} | Errors: {errors:5d}/{total_bits}")
        
    return snr_vec, ber_results

def test_rmse(system, num_trials=100):
    print("\n--- Testing RMSE (Zadoff-Chu, 30.72MHz) ---")
    print(f"Running {num_trials} trials per SNR...")
    
    snr_vec = list(range(13, 21))
    rmse_results = []
    
    for snr in snr_vec:
        rmse_sq_sum = 0
        valid_count = 0
        
        for trial in range(num_trials):
            i_tx = np.random.randint(0, Q)
            d_hat, theta_hat, d_true, theta_true_val, _, _ = system.run_rmse_trial(i_tx, snr, n_sequences=5)
            
            # Using paper Eq(18) logic
            x_est = d_hat * np.cos(theta_hat)
            y_est = d_hat * np.sin(theta_hat)
            
            x_true = d_true * np.cos(theta_true_val)
            y_true = d_true * np.sin(theta_true_val)
            
            rmse_sq = (x_est - x_true)**2 + (y_est - y_true)**2
            rmse_sq_sum += rmse_sq
            valid_count += 1
            
        avg_rmse = np.sqrt(rmse_sq_sum / valid_count)
        rmse_results.append(avg_rmse)
        print(f"SNR = {snr:2d} dB | RMSE = {avg_rmse:.2f} m")
        
    return snr_vec, rmse_results

if __name__ == "__main__":
    np.random.seed(42)
    start_time = time.time()
    
    system = OFDMISACSystem(seq_type='ZC', BW=184.32e6)
    
    # test_ber(system, num_trials=1000)
    test_rmse(system, num_trials=100)
    
    print(f"\nTotal test time: {time.time() - start_time:.1f} seconds")
