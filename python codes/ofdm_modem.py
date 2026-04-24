"""
OFDM Transmitter and Receiver Module
Implements OFDM modulation and demodulation with cyclic prefix
Reference: Paper Eq. (3)-(9) and MATLAB functions
"""

import numpy as np
from system_params import *


class OFDMTransmitter:
    """OFDM Transmitter Implementation"""
    
    def __init__(self, NFFT=NFFT, NCP=NCP):
        """
        Initialize OFDM transmitter
        
        Args:
            NFFT: FFT size
            NCP: Cyclic prefix length
        """
        self.NFFT = NFFT
        self.NCP = NCP
    
    def modulate(self, s_i, Ns):
        """
        OFDM modulation of orthogonal sequence
        
        Process:
        1. DFT of sequence (Eq. 3): S(k) = sum(s_i[n] * exp(-j*2*pi*k*n/Ns))
        2. Map to subcarriers with zero-padding
        3. IFFT (Eq. 4): x(m) = (1/NFFT) * sum(S(k) * exp(j*2*pi*k*m/NFFT))
        4. Insert cyclic prefix (Eq. 5): x_CP(u)
        
        Args:
            s_i: Input orthogonal sequence (Ns,)
            Ns: Sequence length
        
        Returns:
            x_CP: OFDM signal with cyclic prefix (NFFT + NCP,)
        """
        # Ensure sequence is 1D
        s_i = np.atleast_1d(s_i).flatten()
        
        # Step 1: DFT of sequence (Eq. 3)
        S_freq = np.fft.fft(s_i, n=Ns)
        
        # Step 2: Map to NFFT subcarriers with zero-padding
        S_map = np.zeros(self.NFFT, dtype=complex)
        S_map[:Ns] = S_freq
        
        # Step 3: IFFT (Eq. 4)
        x = np.fft.ifft(S_map, n=self.NFFT)
        
        # Step 4: Insert cyclic prefix (Eq. 5)
        # x_CP(u) = x[(u - NCP) mod NFFT]
        x_CP = np.concatenate([x[-self.NCP:], x])
        
        return x_CP


class OFDMReceiver:
    """OFDM Receiver Implementation"""
    
    def __init__(self, NFFT=NFFT, NCP=NCP):
        """
        Initialize OFDM receiver
        
        Args:
            NFFT: FFT size
            NCP: Cyclic prefix length
        """
        self.NFFT = NFFT
        self.NCP = NCP
    
    def demodulate(self, r_sig, Ns, J):
        """
        OFDM demodulation
        
        Process:
        1. Remove cyclic prefix
        2. FFT: R_freq = FFT(r_sig)
        3. Extract used subcarriers: R_sub = R_freq[0:Ns]
        4. IDFT: y_j(n) = IDFT(R_sub)
        
        Args:
            r_sig: Received signal (J, NFFT + NCP)
            Ns: Sequence length
            J: Number of antennas
        
        Returns:
            y_all: Received sequences (J, Ns)
        """
        y_all = np.zeros((J, Ns), dtype=complex)
        
        for jj in range(J):
            # Remove cyclic prefix
            r_nocp = r_sig[jj, self.NCP:self.NCP + self.NFFT]
            
            # FFT
            R_freq = np.fft.fft(r_nocp, n=self.NFFT)
            
            # Extract used subcarriers
            R_sub = R_freq[:Ns]
            
            # IDFT to get time-domain sequence
            y_all[jj, :] = np.fft.ifft(R_sub, n=Ns)
        
        return y_all


if __name__ == "__main__":
    # Test OFDM modulation and demodulation
    print("Testing OFDM Modulation and Demodulation...")
    
    from orthogonal_sequences import initialize_sequences
    
    # Initialize sequences
    seq_dict = initialize_sequences()
    seq_info = seq_dict['ZC']
    s0 = seq_info['s0']
    Ns = seq_info['Ns']
    
    # Create transmitter and receiver
    tx = OFDMTransmitter(NFFT=NFFT, NCP=NCP)
    rx = OFDMReceiver(NFFT=NFFT, NCP=NCP)
    
    # Test modulation
    x_CP = tx.modulate(s0, Ns)
    print(f"\nOFDM Modulation Test:")
    print(f"  Input sequence length: {len(s0)}")
    print(f"  OFDM signal length (with CP): {len(x_CP)}")
    print(f"  Expected length: {NFFT + NCP}")
    print(f"  Signal power: {np.mean(np.abs(x_CP)**2):.4f}")
    
    # Test demodulation (without channel)
    # Simulate single antenna reception
    r_sig = x_CP[np.newaxis, :]  # Convert to (1, N) for single antenna
    y_all = rx.demodulate(r_sig, Ns, J=1)
    
    print(f"\nOFDM Demodulation Test (without channel):")
    print(f"  Received sequence shape: {y_all.shape}")
    print(f"  Input sequence power: {np.mean(np.abs(s0)**2):.4f}")
    print(f"  Received sequence power: {np.mean(np.abs(y_all[0])**2):.4f}")
    print(f"  Correlation with original: {np.abs(np.dot(y_all[0].conj(), s0)) / len(s0):.4f}")
