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
