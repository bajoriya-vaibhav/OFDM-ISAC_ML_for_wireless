"""
Orthogonal Sequence Generation Module
Implements: Zadoff-Chu, m-sequences, and Gold sequences
Reference: Paper Sec. III and MATLAB code
"""

import numpy as np
from system_params import *


def generate_mseq(n, poly_exponents):
    """
    Generate m-sequence of length N = 2^n - 1
    
    Args:
        n: Polynomial order
        poly_exponents: List of exponents for polynomial feedback taps
    
    Returns:
        mseq: m-sequence with values in {-1, +1}
    """
    N = 2**n - 1
    reg = np.ones(n, dtype=int)
    mseq = np.zeros(N, dtype=int)
    
    # Convert exponents to tap positions
    exponents = [e for e in poly_exponents if e < n]
    tap_positions = [n - e for e in exponents]
    tap_positions.append(n)
    
    for k in range(N):
        mseq[k] = reg[-1]
        # XOR feedback
        fb = np.sum(reg[np.array(tap_positions) - 1]) % 2
        reg = np.concatenate([[fb], reg[:-1]])
    
    # Convert to {-1, +1}
    mseq = 2 * mseq - 1
    return mseq.astype(complex)


def generate_zadoff_chu(N, u=1):
    """
    Generate Zadoff-Chu sequence
    
    Args:
        N: Sequence length
        u: Root index (typically u=1)
    
    Returns:
        s0: Zadoff-Chu sequence
    """
    n = np.arange(N)
    s0 = np.exp(-1j * np.pi * u * n * (n + 1) / N)
    return s0


def generate_gold_sequence(n):
    """
    Generate Gold sequence as XOR of two preferred-pair m-sequences
    
    Args:
        n: Polynomial order
    
    Returns:
        gold_seq: Gold sequence with values in {-1, +1}
    """
    # Preferred pair m-sequences for n=10
    m1 = generate_mseq(n, [n, n-3])  # x^10 + x^7 + 1
    m2 = generate_mseq(n, [n, n-7])  # x^10 + x^3 + 1
    
    # XOR operation and convert to {-1, +1}
    gold_bin = np.mod(((m1.real + 1) // 2) + ((m2.real + 1) // 2), 2)
    gold_seq = 2 * gold_bin - 1
    
    return gold_seq.astype(complex)


def make_candidate_sequences(s0, NCS, Q):
    """
    Generate Q candidate sequences by cyclic shifting s0
    Equation (1) in paper: S = {s0, s_NCS, ..., s_{(Q-1)NCS}}
    
    Args:
        s0: Base sequence
        NCS: Cyclic shift spacing (Cyclic Shift Distance)
        Q: Number of candidate sequences
    
    Returns:
        S: (Q, Ns) array of candidate sequences
    """
    Ns = len(s0)
    S = np.zeros((Q, Ns), dtype=complex)
    
    for i in range(Q):
        # Cyclic shift by i * NCS
        S[i, :] = np.roll(s0, i * NCS)
    
    return S


def generate_gray_code(q_bits):
    """
    Generate Gray code mapping for 64-PSK modulation
    
    Args:
        q_bits: Number of bits
    
    Returns:
        gray_enc: Gray code values
        idx_to_bits: Mapping from index to bits
    """
    Q = 2**q_bits
    gray_enc = np.zeros(Q, dtype=int)
    
    # Generate Gray code: gray(i) = i XOR floor(i/2)
    for i in range(Q):
        gray_enc[i] = i ^ (i // 2)
    
    # Create mapping from index to bits
    idx_to_bits = np.zeros((Q, q_bits), dtype=int)
    for i in range(Q):
        # Convert Gray code to binary (MSB first)
        g = gray_enc[i]
        idx_to_bits[i, :] = np.array([(g >> (q_bits - 1 - j)) & 1 for j in range(q_bits)])
    
    return gray_enc, idx_to_bits


def initialize_sequences():
    """
    Initialize all three sequence types with their parameters
    
    Returns:
        sequences_dict: Dictionary containing all sequence information
    """
    sequences_dict = {}
    
    # Zadoff-Chu sequence
    s0_ZC = generate_zadoff_chu(Ns_ZC, u=1)
    S_ZC = make_candidate_sequences(s0_ZC, NCS_ZC, Q)
    sequences_dict['ZC'] = {
        's0': s0_ZC,
        'S': S_ZC,
        'Ns': Ns_ZC,
        'NCS': NCS_ZC,
        'name': 'Zadoff-Chu'
    }
    
    # m-sequence
    s0_mseq = generate_mseq(10, [10, 7])
    S_mseq = make_candidate_sequences(s0_mseq, NCS_mG, Q)
    sequences_dict['mseq'] = {
        's0': s0_mseq,
        'S': S_mseq,
        'Ns': Ns_mG,
        'NCS': NCS_mG,
        'name': 'm-sequence'
    }
    
    # Gold sequence
    s0_gold = generate_gold_sequence(10)
    S_gold = make_candidate_sequences(s0_gold, NCS_mG, Q)
    sequences_dict['gold'] = {
        's0': s0_gold,
        'S': S_gold,
        'Ns': Ns_mG,
        'NCS': NCS_mG,
        'name': 'Gold'
    }
    
    return sequences_dict


if __name__ == "__main__":
    # Test sequence generation
    print("Testing Orthogonal Sequence Generation...")
    
    seq_dict = initialize_sequences()
    
    for key, seq_info in seq_dict.items():
        print(f"\n{seq_info['name']} Sequence:")
        print(f"  Length (Ns): {seq_info['Ns']}")
        print(f"  Cyclic Shift (NCS): {seq_info['NCS']}")
        print(f"  Candidates (Q): {seq_info['S'].shape[0]}")
        print(f"  Base sequence power: {np.mean(np.abs(seq_info['s0'])**2):.4f}")
    
    # Test Gray code
    gray_enc, idx_to_bits = generate_gray_code(q_bits)
    print(f"\nGray Code Mapping (first 10 indices):")
    for i in range(10):
        print(f"  Index {i:2d} -> Gray {gray_enc[i]:2d} -> Bits {idx_to_bits[i]}")
