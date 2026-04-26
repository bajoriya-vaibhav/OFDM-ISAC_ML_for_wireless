"""
System Parameters for OFDM-ISAC System
Reference: Off-the-Shelf OFDM-ISAC System with Clustering Algorithm for Multipath Environments
"""

# Physical layer parameters
d0 = 40.0              # UE-BS distance [m]
theta_true = 45 * 3.14159265359 / 180  # True AoA [rad]
fc = 1.8e9             # Center frequency [Hz]
c_light = 3e8          # Speed of light [m/s]
lambda_wave = c_light / fc  # Wavelength [m]

# OFDM parameters
NFFT = 2048            # FFT size
NCP = 256              # CP length
J = 16                 # BS antennas (ULA)
r_ant = lambda_wave / 2  # Antenna spacing

# Channel parameters
alpha_pl = 2           # Path loss exponent
L = 10                 # Total multipath taps
dtheta_max = 10 * 3.14159265359 / 180  # Max angle deviation [rad]

# Multipath composition
L_interf = round(0.2 * L)   # = 2 interference taps
L_local = L - L_interf       # = 8 locally scattered taps

# Index modulation parameters
q_bits = 6             # 6-bit transmission
Q = 2 ** q_bits        # 64 candidate sequences

# Orthogonal Sequence parameters
# Zadoff-Chu sequence
Ns_ZC = 839
NCS_ZC = 13

# m-sequence / Gold sequence
Ns_mG = 1023
NCS_mG = 15

# BER Simulation parameters
SNR_dB_min = 13
SNR_dB_max = 20
SNR_dB_step = 1
num_trials_BER = 10000  # Monte Carlo trials

# RMSE Simulation parameters
bandwidth_options = [30.72e6, 61.44e6, 184.32e6]  # Bandwidth options [Hz]
num_trials_RMSE = 10000  # Monte Carlo trials for RMSE

# DBSCAN clustering parameters
eta_threshold = 0.5    # Power threshold for clustering
epsilon = 0.2          # DBSCAN radius parameter
min_samples = 3        # DBSCAN minimum samples parameter

# Gray mapping parameters
gray_mapping = True    # Use Gray mapping for 64-PSK

# Multi-user (MU) parameters
NUM_UES = 3                          # Number of UEs
UE_DISTANCES = [40.0, 55.0, 30.0]   # Per-UE distances [m]
UE_ANGLES_DEG = [45.0, 120.0, 75.0]  # Per-UE AoA [degrees]
UE_ANGLES_RAD = [a * 3.14159265359 / 180 for a in UE_ANGLES_DEG]
UE_ZC_ROOTS = [1, 3, 7]             # Per-UE Zadoff-Chu root indices

# Simulation control
verbose = True         # Print simulation progress

# Mobile UE parameters
v_ue = 30.0                          # UE velocity [m/s] (~108 km/h)
BW_default = 30.72e6                 # Default bandwidth for T_sym calc [Hz]
T_sym = (NFFT + NCP) / BW_default   # OFDM symbol duration [s] ≈ 75 μs
M_doppler = 32                       # Slow-time symbols for Doppler FFT
