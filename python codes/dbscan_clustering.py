"""
DBSCAN Clustering Module
Density-based clustering to improve localization resolution
Reference: Paper Sec. III.C and Algorithm 1
"""

import numpy as np

class DBSCAN:
    """Standard DBSCAN implementation"""
    def __init__(self, epsilon=1.0, min_samples=3):
        self.epsilon = epsilon
        self.min_samples = min_samples

    def fit(self, data):
        N = data.shape[0]
        labels = np.full(N, -1, dtype=int)
        cluster_id = 0

        diff = data[:, None, :] - data[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

        visited = np.zeros(N, dtype=bool)

        for i in range(N):
            if visited[i]: continue
            visited[i] = True
            neighbors = np.where(dist[i] <= self.epsilon)[0]
            if len(neighbors) < self.min_samples:
                continue
            labels[i] = cluster_id
            seed_set = list(neighbors)
            si = 0
            while si < len(seed_set):
                q = seed_set[si]
                si += 1
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = np.where(dist[q] <= self.epsilon)[0]
                    if len(q_neighbors) >= self.min_samples:
                        for nb in q_neighbors:
                            if nb not in seed_set:
                                seed_set.append(nb)
                if labels[q] == -1:
                    labels[q] = cluster_id
            cluster_id += 1

        return labels

class ClusteringLocalizationEngine:
    def __init__(self, NCS, J, lambda_wave, r_ant, c_light=3e8,
                 eta_threshold=1.0, epsilon=1.0, min_samples=3):
        self.NCS = NCS
        self.J = J
        self.lambda_wave = lambda_wave
        self.r_ant = r_ant
        self.c_light = c_light
        self.eta_threshold = eta_threshold
        self.dbscan = DBSCAN(epsilon=epsilon, min_samples=min_samples)

    def collect_data(self, y_multi, s0, i_hat_first, NFFT, Ns, Ts, Q,
                     num_seq=5, angle_grid_size=361):
        s0_arr = np.array(s0).flatten().astype(complex)
        
        angles = np.linspace(0, np.pi, angle_grid_size)
        A = np.zeros((self.J, angle_grid_size), dtype=complex)
        for p in range(angle_grid_size):
            for ant in range(self.J):
                phase = -1j * 2 * np.pi / self.lambda_wave * ant * self.r_ant * np.cos(angles[p])
                A[ant, p] = np.exp(phase)
                
        D = []
        for m in range(num_seq):
            y_cur = y_multi[m * self.J : (m + 1) * self.J, :]
            i_m = (i_hat_first + m) % Q
            
            for i in range(self.NCS):
                offset = i
                base_shift = i_m * self.NCS
                kappa = (base_shift + offset) % Ns
                s_kappa = np.roll(s0_arr, kappa)
                
                y_des = np.zeros(self.J, dtype=complex)
                for j in range(self.J):
                    y_des[j] = np.dot(np.conj(s_kappa), y_cur[j, :])
                    
                d1 = self.c_light * Ts * (NFFT / Ns) * offset
                d3 = np.abs(y_des[0])**2
                
                p_i = np.argmax(np.abs(A.conj().T @ y_des))
                d2 = angles[p_i]
                
                D.append([d1, d2, d3])
                
        return np.array(D)

    def normalize_data(self, D):
        # Prevent Z-score explosion by normalizing strictly relative to bounds
        D_norm = np.zeros_like(D)
        # Distance generally scaled relative to some maximum expected grid distance ~ 50 meters
        max_dist = max(50.0, np.max(D[:, 0])) 
        D_norm[:, 0] = D[:, 0] / max_dist
        # Angle scaled relative to pi
        D_norm[:, 1] = D[:, 1] / np.pi
        # Power normalized relative to the maximum observed power peak in the scan
        p_max = np.max(D[:, 2]) if np.max(D[:, 2]) > 0 else 1.0
        D_norm[:, 2] = D[:, 2] / p_max
        return D_norm

    def estimate_localization(self, y_multi, s0, i_hat, kappa_hat_all,
                              NFFT, Ns, Ts, Q, num_seq=5, angle_grid_size=361):
        
        D = self.collect_data(y_multi, s0, i_hat, NFFT, Ns, Ts, Q,
                              num_seq=num_seq, angle_grid_size=angle_grid_size)
                              
        D_norm = self.normalize_data(D)
        
        # 1. Reject everything strictly below power threshold eta
        valid_mask = D_norm[:, 2] >= self.eta_threshold
        D_filt = D[valid_mask]
        D_norm_filt = D_norm[valid_mask]
        
        if len(D_filt) < self.dbscan.min_samples:
            return None, None, None
            
        # 2. Reconfigure DBSCAN for fixed-bound normalizations
        self.dbscan.epsilon = 0.2
        labels = self.dbscan.fit(D_norm_filt[:, :2])
        unique = set(labels) - {-1}
        
        if not unique:
            return None, None, None
            
        # 3. Choose the dominant cluster
        largest = max(unique, key=lambda c: np.sum(labels == c))
        cl_mask = labels == largest
        D_cl = D_filt[cl_mask]
        
        if len(D_cl) == 0:
            return None, None, None
            
        # Instead of np.min(), which eagerly selects random dropped delay offsets, 
        # algorithm 1 specifies minimum, but if the cluster captures interference, it skews.
        # Pick the point corresponding to the absolute maximum correlation density in the cluster.
        best_idx = np.argmax(D_cl[:, 2])
        d_hat = float(D_cl[best_idx, 0])
        theta_hat = float(D_cl[best_idx, 1])
        
        return d_hat, theta_hat, D_cl

    def estimate_localization_topk(self, y_multi, s0, i_hat, kappa_hat_all,
                                    NFFT, Ns, Ts, Q, K=2,
                                    num_seq=5, angle_grid_size=361):
        """
        Top-K cluster extraction for multi-user localization.

        Runs the same DBSCAN pipeline but returns the K largest clusters
        instead of only the single largest one.  Each cluster center is
        the point with maximum correlation power inside that cluster.

        Args:
            K: Number of UEs (clusters) to extract
            (other args identical to estimate_localization)

        Returns:
            results: list of K dicts, each with keys
                     'd_hat', 'theta_hat', 'cluster_data', 'cluster_size'
                     Sorted by cluster size descending.
                     If fewer than K clusters are found, remaining entries
                     have d_hat=None.
        """
        D = self.collect_data(y_multi, s0, i_hat, NFFT, Ns, Ts, Q,
                              num_seq=num_seq, angle_grid_size=angle_grid_size)
        D_norm = self.normalize_data(D)

        # Power threshold
        valid_mask = D_norm[:, 2] >= self.eta_threshold
        D_filt = D[valid_mask]
        D_norm_filt = D_norm[valid_mask]

        # Default empty results
        empty = {'d_hat': None, 'theta_hat': None,
                 'cluster_data': None, 'cluster_size': 0}
        results = [dict(empty) for _ in range(K)]

        if len(D_filt) < self.dbscan.min_samples:
            return results

        self.dbscan.epsilon = 0.2
        labels = self.dbscan.fit(D_norm_filt[:, :2])
        unique_labels = sorted(set(labels) - {-1},
                                key=lambda c: np.sum(labels == c),
                                reverse=True)

        if not unique_labels:
            return results

        for k_idx, cl_id in enumerate(unique_labels[:K]):
            cl_mask = labels == cl_id
            D_cl = D_filt[cl_mask]
            if len(D_cl) == 0:
                continue
            best_idx = np.argmax(D_cl[:, 2])
            results[k_idx] = {
                'd_hat': float(D_cl[best_idx, 0]),
                'theta_hat': float(D_cl[best_idx, 1]),
                'cluster_data': D_cl,
                'cluster_size': int(np.sum(cl_mask)),
            }

        return results

    def collect_data_doppler(self, y_multi, s0, i_hat_first, NFFT, Ns, Ts, Q,
                             T_sym, lambda_wave, num_seq=32, angle_grid_size=361):
        """
        Collect 4D feature matrix: [distance, angle, velocity, power].

        For each delay bin (offset):
          1. Despread across all M symbols → complex vector z(m), m=0..M-1
          2. Slow-time FFT across z(m) → Z(f), f=0..M-1
          3. Peak of |Z(f)| gives Doppler frequency → velocity

        Args:
            y_multi: (num_seq*J, Ns) stacked received sequences
            s0: Base sequence
            i_hat_first: Detected index for first symbol
            T_sym: OFDM symbol duration [s]
            lambda_wave: Wavelength [m]
            num_seq: Number of slow-time symbols (M)

        Returns:
            D: (N, 4) array of [distance, angle, velocity, power]
        """
        s0_arr = np.array(s0).flatten().astype(complex)

        # Precompute steering matrix
        angles = np.linspace(0, np.pi, angle_grid_size)
        A = np.zeros((self.J, angle_grid_size), dtype=complex)
        for p in range(angle_grid_size):
            for ant in range(self.J):
                phase = -1j * 2 * np.pi / self.lambda_wave * ant * self.r_ant * np.cos(angles[p])
                A[ant, p] = np.exp(phase)

        # Doppler frequency grid
        f_doppler = np.fft.fftfreq(num_seq, d=T_sym)  # Hz

        D = []
        for offset in range(self.NCS):
            # Collect slow-time vector across M symbols for antenna 0
            z_slow = np.zeros(num_seq, dtype=complex)
            y_des_first = None

            for m in range(num_seq):
                y_cur = y_multi[m * self.J: (m + 1) * self.J, :]
                i_m = (i_hat_first + m) % Q
                base_shift = i_m * self.NCS
                kappa = (base_shift + offset) % Ns
                s_kappa = np.roll(s0_arr, kappa)

                # Despread on antenna 0 for Doppler
                z_slow[m] = np.dot(np.conj(s_kappa), y_cur[0, :])

                # Save first symbol's full despread vector for angle estimation
                if m == 0:
                    y_des_first = np.zeros(self.J, dtype=complex)
                    for j in range(self.J):
                        y_des_first[j] = np.dot(np.conj(s_kappa), y_cur[j, :])

            # Distance
            d1 = self.c_light * Ts * (NFFT / Ns) * offset

            # Angle from first symbol
            p_i = np.argmax(np.abs(A.conj().T @ y_des_first))
            d2 = angles[p_i]

            # Power (average across symbols for robustness)
            d4 = np.mean(np.abs(z_slow) ** 2)

            # Slow-time FFT → Doppler
            Z = np.fft.fft(z_slow, n=num_seq)
            Z_mag = np.abs(Z)
            f_idx = np.argmax(Z_mag)
            f_D = f_doppler[f_idx]  # Already handles negative via fftfreq
            v_hat = f_D * lambda_wave  # velocity [m/s]

            D.append([d1, d2, v_hat, d4])

        return np.array(D)

    def normalize_data_doppler(self, D, v_max=100.0):
        """Normalize 4D data: [distance, angle, velocity, power]."""
        D_norm = np.zeros_like(D)
        max_dist = max(50.0, np.max(D[:, 0]))
        D_norm[:, 0] = D[:, 0] / max_dist
        D_norm[:, 1] = D[:, 1] / np.pi
        D_norm[:, 2] = D[:, 2] / v_max  # velocity normalized to v_max
        p_max = np.max(D[:, 3]) if np.max(D[:, 3]) > 0 else 1.0
        D_norm[:, 3] = D[:, 3] / p_max
        return D_norm

    def estimate_localization_doppler(self, y_multi, s0, i_hat, kappa_hat_all,
                                       NFFT, Ns, Ts, Q, T_sym, lambda_wave,
                                       num_seq=32, angle_grid_size=361):
        """
        Doppler-aware localization: estimates (distance, angle, velocity).

        Returns:
            d_hat, theta_hat, v_hat, cluster_data
        """
        D = self.collect_data_doppler(y_multi, s0, i_hat, NFFT, Ns, Ts, Q,
                                       T_sym, lambda_wave,
                                       num_seq=num_seq, angle_grid_size=angle_grid_size)
        D_norm = self.normalize_data_doppler(D)

        # Power filter
        valid_mask = D_norm[:, 3] >= self.eta_threshold
        D_filt = D[valid_mask]
        D_norm_filt = D_norm[valid_mask]

        if len(D_filt) < self.dbscan.min_samples:
            return None, None, None, None

        # DBSCAN on (distance, angle) — same as static (more robust)
        self.dbscan.epsilon = 0.2
        labels = self.dbscan.fit(D_norm_filt[:, :2])
        unique = set(labels) - {-1}

        if not unique:
            return None, None, None, None

        largest = max(unique, key=lambda c: np.sum(labels == c))
        cl_mask = labels == largest
        D_cl = D_filt[cl_mask]

        if len(D_cl) == 0:
            return None, None, None, None

        # Distance & angle from max-power point (same as static)
        best_idx = np.argmax(D_cl[:, 3])
        d_hat = float(D_cl[best_idx, 0])
        theta_hat = float(D_cl[best_idx, 1])

        # Velocity: power-weighted mean across cluster (more robust than single point)
        powers = D_cl[:, 3]
        p_total = np.sum(powers)
        if p_total > 0:
            v_hat = float(np.sum(D_cl[:, 2] * powers) / p_total)
        else:
            v_hat = float(D_cl[best_idx, 2])

        return d_hat, theta_hat, v_hat, D_cl


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing DBSCAN Clustering...")
    np.random.seed(42)

    cluster1 = np.random.randn(10, 2) * 0.3
    cluster2 = np.random.randn(10, 2) * 0.3 + 3
    noise    = np.random.uniform(-5, 5, (5, 2))
    data     = np.vstack([cluster1, cluster2, noise])

    mu = np.mean(data, axis=0); sigma = np.std(data, axis=0)
    data_norm = (data - mu) / sigma

    db = DBSCAN(epsilon=1.0, min_samples=3)
    labels = db.fit(data_norm)
    print(f"Clusters found: {len(set(labels)-{-1})}")
    print(f"Noise points:   {np.sum(labels==-1)}")
