## OFDM-ISAC Complete Simulation

Key equations implemented:
Eq.(1): Candidate sequence generation via cyclic shift
Eq.(3)-(5): OFDM TX (DFT -> subcarrier map -> IFFT -> CP)
Eq.(6): Tapped delay line channel model
Eq.(8)-(9): OFDM RX (CP removal -> FFT -> demap -> IDFT)
Eq.(11): Correlation profile
Eq.(12): Sequence detection
Eq.(14): Distance estimation
Eq.(15)-(16): Direction estimation (despreading + spatial correlation)
Algorithm 1: DBSCAN data collection
Eq.(17): Clustering-based localization
Eq.(18): RMSE definition