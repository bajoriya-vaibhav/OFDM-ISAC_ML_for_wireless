"""Quick RMSE diagnostic — run 20 trials at SNR=15dB and print each estimate"""
import numpy as np
import sys
sys.path.insert(0, '.')
from ofdm_isac import OFDMISACSystem
from system_params import *

system = OFDMISACSystem(seq_type='ZC', BW=30.72e6)
print(f"BW=30.72MHz, Ns={system.Ns}, NCS={system.NCS}")
print(f"Distance resolution per sample: {3e8 * system.Ts * (NFFT/system.Ns):.2f} m")
print(f"tau0 = round(40 / (c*Ts)) = {round(40 / (3e8 * system.Ts))}")
print()

for snr in [15, 20]:
    d_errs = []
    th_errs = []
    for trial in range(20):
        i_tx = np.random.randint(0, Q)
        d_hat, theta_hat, d_true, theta_true_val, _, _ = system.run_rmse_trial(i_tx, snr, n_sequences=5)
        d_err = abs(d_hat - d_true)
        th_err = abs(theta_hat - theta_true_val) * 180 / np.pi
        d_errs.append(d_err)
        th_errs.append(th_err)
        if trial < 5:
            print(f"SNR={snr}dB trial {trial}: d_hat={d_hat:.1f}m (err={d_err:.1f}m), "
                  f"theta_hat={theta_hat*180/np.pi:.1f}deg (err={th_err:.1f}deg)")
    
    rmse_d = np.sqrt(np.mean(np.array(d_errs)**2))
    rmse_th = np.sqrt(np.mean(np.array(th_errs)**2))
    print(f"SNR={snr}dB avg over 20: RMSE_d={rmse_d:.2f}m, RMSE_theta={rmse_th:.2f}deg\n")
