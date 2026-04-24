"""
Channel and Noise Modeling Module
Local scattering channel model with multipath and interference
Reference: Paper Sec. II and MATLAB gen_channel() function
"""

import numpy as np
from system_params import *


class ChannelModel:
    """
    Local scattering channel model with multipath components
    """
    
    def __init__(self, d0=d0, theta=theta_true, lambda_wave=lambda_wave,
                 r_ant=r_ant, J=J, L_local=L_local, L_interf=L_interf,
                 dtheta_max=dtheta_max, alpha_pl=alpha_pl):
        """
        Initialize channel model parameters
        
        Args:
            d0: UE-BS distance [m]
            theta: True angle of arrival [rad]
            lambda_wave: Wavelength [m]
            r_ant: Antenna spacing [m]
            J: Number of antennas
            L_local: Number of local scattering paths
            L_interf: Number of interference paths
            dtheta_max: Maximum angle deviation [rad]
            alpha_pl: Path loss exponent
        """
        self.d0 = d0
        self.theta = theta
        self.lambda_wave = lambda_wave
        self.r_ant = r_ant
        self.J = J
        self.L_local = L_local
        self.L_interf = L_interf
        self.L = L_local + L_interf
        self.dtheta_max = dtheta_max
        self.alpha_pl = alpha_pl
    
    def generate_channel(self, Ts, c_light=c_light):
        """
        Generate multipath channel impulse response
        
        Implements Eq. (6) in paper:
        x_CH_l,j(p) = sqrt(d^-alpha) * h_j[l] * x_CP[p - l - tau_j] + z_j(p)
        
        Args:
            Ts: Sampling time
            c_light: Speed of light [m/s]
        
        Returns:
            h_delay: (J, L) channel coefficients for each antenna and path
            tau_int: (L,) propagation delay indices for each path
        """
        h_delay = np.zeros((self.J, self.L), dtype=complex)
        tau_int = np.zeros(self.L, dtype=int)
        
        # Delay for direct path
        tau0 = int(np.round(self.d0 / (c_light * Ts)))
        
        for l in range(self.L):
            # ── Local scattered paths (l < L_local) ──────────────────────────
            # All originate from UE at distance d0 → same delay tau0.
            # Their angle deviates slightly from the true AoA (local scattering).
            # ── Interference paths (l >= L_local) ────────────────────────────
            # Random distant scatterers → random delay offset from tau0.
            if l < self.L_local:
                tau_int[l]  = tau0          # all local paths: same delay
                dl          = self.d0       # same distance
                dtheta      = (2 * np.random.rand() - 1) * self.dtheta_max
                theta_l     = self.theta + dtheta
            else:
                # Interference: random delay 1..NCP-1 samples AWAY from tau0
                tau_int[l]  = tau0 + (l - self.L_local + 1)
                dl          = self.d0 * (1 + 0.1 * np.random.rand())
                theta_l     = (np.random.rand() - 0.5) * np.pi  # random direction
            
            # Complex Rayleigh fading gain
            beta_l = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            
            # ULA phase shift per antenna
            for jj in range(self.J):
                phase = (-1j * 2 * np.pi / self.lambda_wave *
                         jj * self.r_ant * np.cos(theta_l))
                path_loss       = np.sqrt(dl ** (-self.alpha_pl))
                h_delay[jj, l]  = beta_l * np.exp(phase) * path_loss
        
        return h_delay, tau_int


class NoiseModel:
    """
    Additive White Gaussian Noise (AWGN) model
    """
    
    @staticmethod
    def calculate_noise_variance(SNR_lin, NFFT, NCP, Ns):
        """
        Calculate noise variance based on SNR
        
        From MATLAB code:
        noise_var = 5.0 * (NFFT + NCP) / (Ns * SNR_lin)
        
        Args:
            SNR_lin: Linear SNR (not in dB)
            NFFT: FFT size
            NCP: CP length
            Ns: Sequence length
        
        Returns:
            noise_var: Noise variance
        """
        noise_var = 5.0 * (NFFT + NCP) / (Ns * SNR_lin)
        return noise_var
    

    
    @staticmethod
    def add_awgn(signal, noise_var, J):
        """
        Add AWGN to received signal
        
        Args:
            signal: Input signal (J, signal_length)
            noise_var: Noise variance
            J: Number of antennas
        
        Returns:
            noisy_signal: Signal with added noise
        """
        signal_len = signal.shape[1]
        noise = np.sqrt(noise_var / 2) * (np.random.randn(J, signal_len) + 
                                          1j * np.random.randn(J, signal_len))
        return signal + noise


def snr_db_to_linear(SNR_dB):
    """Convert SNR from dB to linear scale"""
    return 10 ** (SNR_dB / 10)


def snr_linear_to_db(SNR_lin):
    """Convert SNR from linear to dB scale"""
    return 10 * np.log10(SNR_lin)


if __name__ == "__main__":
    # Test channel model
    print("Testing Channel Model...")
    
    channel = ChannelModel()
    Ts = 1 / 30.72e6  # Sampling time for 30.72 MHz bandwidth
    
    # Generate channel
    h_delay, tau_int = channel.generate_channel(Ts)
    
    print(f"\nChannel Parameters:")
    print(f"  Distance (d0): {channel.d0} m")
    print(f"  True AoA: {channel.theta * 180 / np.pi:.1f}°")
    print(f"  Number of antennas (J): {channel.J}")
    print(f"  Total paths (L): {channel.L}")
    print(f"  Local scattered paths: {channel.L_local}")
    print(f"  Interference paths: {channel.L_interf}")
    
    print(f"\nGenerated Channel Info:")
    print(f"  Channel shape: {h_delay.shape}")
    print(f"  Max delay index: {np.max(tau_int)}")
    print(f"  Min delay index: {np.min(tau_int)}")
    print(f"  Typical channel power: {np.mean(np.abs(h_delay)**2):.4f}")
    
    # Test noise model
    print(f"\nTesting Noise Model...")
    SNR_dB = 15
    SNR_lin = snr_db_to_linear(SNR_dB)
    noise_var = NoiseModel.calculate_noise_variance(SNR_lin, NFFT, NCP, 839)
    print(f"  SNR: {SNR_dB} dB ({SNR_lin:.2f} linear)")
    print(f"  Noise variance: {noise_var:.6f}")
