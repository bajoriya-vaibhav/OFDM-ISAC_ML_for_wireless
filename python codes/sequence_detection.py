"""
Sequence Detection and Localization Module
Implements sequence detection, distance and direction estimation
Reference: Paper Sec. III.A, III.B and MATLAB detect_seq() function
"""

import numpy as np
from system_params import *


class SequenceDetector:
    """Sequence detection and demodulation"""
    
    def __init__(self, J=J):
        """
        Initialize sequence detector
        
        Args:
            J: Number of antennas
        """
        self.J = J
    
    def detect_sequence(self, y_all, s0, NCS, Q, use_diversity=True):
        """
        Detect transmitted sequence index using autocorrelation
        
        Process:
        1. For each antenna, calculate correlation with s0 (Eq. 11)
        2. Find peak index kappa_hat (Eq. 12)
        3. Estimate sequence index as floor(kappa_hat / NCS)
        4. Use mode (majority voting) across antennas for diversity
        
        Args:
            y_all: Received sequences (J, Ns)
            s0: Base orthogonal sequence (Ns,)
            NCS: Cyclic shift spacing
            Q: Number of candidate sequences
            use_diversity: Use spatial diversity with multiple antennas
        
        Returns:
            i_hat: Estimated sequence index
            kappa_hat_all: Peak correlation indices for all antennas (J,)
        """
        Ns = len(s0)
        i_per_ant = np.zeros(self.J, dtype=int)
        kappa_per_ant = np.zeros(self.J, dtype=int)
        
        # FFT of base sequence for frequency domain correlation
        S0_fft = np.fft.fft(s0, n=Ns)
        
        for jj in range(self.J):
            # Received sequence for antenna jj
            Yj = np.fft.fft(y_all[jj, :], n=Ns)
            
            # Correlation in frequency domain (matched filter)
            # Eq. (11): R_j(kappa) = sum(y_j[n] * conj(s0[kappa + n]))
            # Implemented efficiently in frequency domain via IFFT
            R_j = np.abs(np.fft.ifft(Yj * np.conj(S0_fft), n=Ns))
            
            # Find peak index
            kappa_hat = np.argmax(R_j)
            kappa_per_ant[jj] = kappa_hat
            
            # Estimate sequence index: i_j = floor(kappa_hat / NCS)
            # Eq. (12): i_hat_j = floor(kappa_hat_j / NCS)
            i_j = int(np.floor(kappa_hat / NCS))
            i_j = min(i_j, Q - 1)  # Ensure valid index
            i_per_ant[jj] = i_j
        
        # Final detection: use spatial diversity
        if use_diversity and self.J > 1:
            # Use mode (most frequent index) across antennas
            i_hat = int(np.median(i_per_ant))  # Alternative: use mode
            # For true mode, use scipy.stats.mode or custom implementation
            unique, counts = np.unique(i_per_ant, return_counts=True)
            i_hat = unique[np.argmax(counts)]
        else:
            i_hat = i_per_ant[0]
        
        return i_hat, kappa_per_ant
    
    def demodulate(self, i_hat, gray_enc, idx_to_bits):
        """
        Demodulate sequence index to bits
        
        Args:
            i_hat: Detected sequence index
            gray_enc: Gray code table
            idx_to_bits: Index to bits mapping
        
        Returns:
            s_hat: Estimated sequence (not used in our implementation)
            b_hat: Estimated bits
        """
        i_hat = int(np.clip(i_hat, 0, len(gray_enc) - 1))
        b_hat = idx_to_bits[i_hat, :]
        return None, b_hat


class LocalizationEstimator:
    """Distance and direction estimation for localization"""
    
    def __init__(self, J=J, lambda_wave=lambda_wave, r_ant=r_ant, c_light=3e8):
        """
        Initialize localization estimator
        
        Args:
            J: Number of antennas
            lambda_wave: Wavelength
            r_ant: Antenna spacing
            c_light: Speed of light
        """
        self.J = J
        self.lambda_wave = lambda_wave
        self.r_ant = r_ant
        self.c_light = c_light
    
    def estimate_distance(self, kappa_hat, i_hat, NCS, NFFT, Ns, Ts):
        """
        Estimate distance from estimated delay
        
        Eq. (14): d_kappa = c * Ts * (NFFT/Ns) * (kappa_hat - i_hat * NCS)
        
        This calculates distance for each antenna
        
        Args:
            kappa_hat: Peak correlation indices (J,)
            i_hat: Sequence index (scalar)
            NCS: Cyclic shift spacing
            NFFT: FFT size
            Ns: Sequence length
            Ts: Sampling time
        
        Returns:
            d_estimated: Estimated distances for each antenna (J,)
        """
        d_estimated = np.zeros(self.J)
        
        for jj in range(self.J):
            # Eq. (14)
            d_estimated[jj] = self.c_light * Ts * (NFFT / Ns) * (kappa_hat[jj] - i_hat * NCS)
        
        return d_estimated
    
    def estimate_direction(self, y_all, s0, kappa_hat, i_hat, NCS, angle_grid_size=181):
        """
        Estimate direction using correlation method (optimal by Cauchy-Schwarz)
        
        Process:
        1. Despreading: y_des(j) = s_kappa^H * y_j (Eq. 15)
        2. Create steering vectors for P potential angles
        3. Direction finding: p_hat = argmax |a_p^H * y_des| (Eq. 16)
        
        Args:
            y_all: Received sequences (J, Ns)
            s0: Base orthogonal sequence
            kappa_hat: Peak correlation indices (J,)
            i_hat: Sequence index
            NCS: Cyclic shift spacing
            angle_grid_size: Number of angle grid points (default 181 for 0-180 degrees)
        
        Returns:
            theta_estimated: Estimated direction of arrival (J,)
        """
        Ns = len(s0)
        theta_estimated = np.zeros(self.J)
        
        # Generate candidate sequences by cyclic shift
        s0_array = np.array(s0)
        
        for jj in range(self.J):
            # Get the cyclic shift index
            kappa_j = int(kappa_hat[jj])
            shift_amount = kappa_j - i_hat * NCS
            shift_amount = int(np.clip(shift_amount, 0, Ns - 1))
            
            # Create the shifted sequence for despreading
            s_shifted = np.roll(s0_array, shift_amount)
            
            # Eq. (15): y_des(j) = s_H * y_j
            y_des = np.dot(np.conj(s_shifted), y_all[jj, :])
            
            # Create spatial steering dictionary (ULA)
            angles = np.linspace(-np.pi/2, np.pi/2, angle_grid_size)
            correlations = np.zeros(len(angles))
            
            for p, angle in enumerate(angles):
                # Steering vector for angle
                # a(angle) = [1, exp(-j*2*pi/lambda*r*cos(angle)), ...]
                a_p = np.zeros(self.J, dtype=complex)
                for ant in range(self.J):
                    phase = -1j * 2 * np.pi / self.lambda_wave * ant * self.r_ant * np.cos(angle)
                    a_p[ant] = np.exp(phase)
                
                # Correlation with despreading signal
                # Note: We need to gather signals from all antennas
                # For now, using single antenna correlation as approximation
                correlations[p] = np.abs(np.dot(np.conj(a_p), y_all[:, shift_amount]))
            
            # Find best angle
            best_p = np.argmax(correlations)
            theta_estimated[jj] = angles[best_p]
        
        return theta_estimated
    
    def estimate_direction_beamforming(self, y_all, s0, kappa_hat, i_hat, NCS, 
                                       angle_grid_size=181):
        """
        Alternative direction estimation using multi-antenna beamforming
        
        Args:
            y_all: Received sequences (J, Ns)
            s0: Base orthogonal sequence
            kappa_hat: Peak correlation indices (J,)
            i_hat: Sequence index
            NCS: Cyclic shift spacing
            angle_grid_size: Number of angle grid points
        
        Returns:
            theta_estimated: Estimated direction (scalar, averaged across delays)
        """
        Ns = len(s0)
        s0_array = np.array(s0)
        
        # Get dominant delay index
        kappa_dominant = int(kappa_hat[0])  # Use first antenna as reference
        shift_amount = kappa_dominant - i_hat * NCS
        shift_amount = int(np.clip(shift_amount, 0, Ns - 1))
        
        # Despreading signal across all antennas at this delay
        s_shifted = np.roll(s0_array, shift_amount)
        y_des = np.dot(np.conj(s_shifted.reshape(-1, 1)), y_all[:, shift_amount].reshape(1, -1))
        y_des_vector = y_all[:, shift_amount]  # (J,) vector
        
        # Beamforming: sweep angles and find best match
        angles = np.linspace(-np.pi/2, np.pi/2, angle_grid_size)
        power_spectrum = np.zeros(len(angles))
        
        for p, angle in enumerate(angles):
            # Steering vector
            a_p = np.zeros(self.J, dtype=complex)
            for ant in range(self.J):
                phase = -1j * 2 * np.pi / self.lambda_wave * ant * self.r_ant * np.cos(angle)
                a_p[ant] = np.exp(phase)
            
            # Beamformer output
            beam_output = np.abs(np.dot(np.conj(a_p), y_des_vector))
            power_spectrum[p] = beam_output
        
        # Find peak
        best_p = np.argmax(power_spectrum)
        theta_estimated = angles[best_p]
        
        return theta_estimated


if __name__ == "__main__":
    print("Testing Sequence Detection and Localization...")
    
    from orthogonal_sequences import initialize_sequences
    from ofdm_modem import OFDMTransmitter, OFDMReceiver
    from channel_noise import ChannelModel, NoiseModel
    
    # Setup
    seq_dict = initialize_sequences()
    seq_info = seq_dict['ZC']
    s0 = seq_info['s0']
    Ns = seq_info['Ns']
    NCS = seq_info['NCS']
    
    # Create components
    tx = OFDMTransmitter()
    rx = OFDMReceiver()
    channel = ChannelModel()
    detector = SequenceDetector(J=J)
    localizer = LocalizationEstimator(J=J)
    
    # Transmit
    x_CP = tx.modulate(s0, Ns)
    print(f"Transmitted signal length: {len(x_CP)}")
    
    # Channel (simple test without actual channel)
    # For testing, just add small noise to received signal
    noise_var = 0.001
    r_sig = x_CP[np.newaxis, :].repeat(J, axis=0)  # Replicate for J antennas
    r_sig = r_sig + np.sqrt(noise_var/2) * (np.random.randn(J, len(x_CP)) + 
                                             1j * np.random.randn(J, len(x_CP)))
    
    # Receive
    y_all = rx.demodulate(r_sig, Ns, J=J)
    print(f"Received sequences shape: {y_all.shape}")
    
    # Detect
    i_hat, kappa_hat = detector.detect_sequence(y_all, s0, NCS, 64)
    print(f"Detected sequence index: {i_hat}")
    print(f"Peak indices per antenna: {kappa_hat}")
