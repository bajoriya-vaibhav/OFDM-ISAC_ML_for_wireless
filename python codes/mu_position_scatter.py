"""
Multi-User DBSCAN Position Scatter Plot
Runs N RMSE trials at a fixed SNR and plots all per-UE position estimates
in Cartesian (x, y) coordinates, with true positions marked as X.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from system_params import *
from multi_user_isac import MultiUserISACSystem


def mu_position_scatter(mu_system, snr_db=20, num_trials=100, n_sequences=5):
    """
    Collect per-UE position estimates and plot scatter.

    Args:
        mu_system: MultiUserISACSystem instance
        snr_db: Fixed SNR to evaluate at [dB]
        num_trials: Number of RMSE trials to run
        n_sequences: Symbols per trial (M)

    Returns:
        per_ue_points: dict {k: {'x_est': [], 'y_est': [], 'x_true': float, 'y_true': float}}
    """
    K = mu_system.K
    per_ue_points = {}
    for k in range(K):
        d_true = mu_system.per_ue[k]['d0']
        theta_true = mu_system.per_ue[k]['theta']
        per_ue_points[k] = {
            'x_est': [], 'y_est': [],
            'x_true': d_true * np.cos(theta_true),
            'y_true': d_true * np.sin(theta_true),
        }

    print(f"Collecting {num_trials} position estimates at SNR={snr_db} dB ...")

    for trial in range(num_trials):
        i_tx_list = [np.random.randint(0, Q) for _ in range(K)]
        results = mu_system.run_rmse_trial(i_tx_list, snr_db,
                                            n_sequences=n_sequences)

        for k in range(K):
            r = results[k]
            if r['d_hat'] is not None and np.isfinite(r['d_hat']):
                x_e = r['d_hat'] * np.cos(r['theta_hat'])
                y_e = r['d_hat'] * np.sin(r['theta_hat'])
                per_ue_points[k]['x_est'].append(x_e)
                per_ue_points[k]['y_est'].append(y_e)

        if (trial + 1) % 25 == 0:
            print(f"  Trial {trial+1}/{num_trials} done")

    return per_ue_points


def plot_scatter(per_ue_points, K, snr_db, save_path='./mu_fig_position_scatter.png'):
    """Generate the scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 8))

    ue_colors = ['#4a90d9', '#e8923f', '#5cb85c', '#d9534f']
    ue_labels = []

    for k in range(K):
        pts = per_ue_points[k]
        x_arr = np.array(pts['x_est'])
        y_arr = np.array(pts['y_est'])

        ax.scatter(x_arr, y_arr, c=ue_colors[k % len(ue_colors)],
                   alpha=0.35, s=30, edgecolors='none',
                   label=f'User {k+1} estimates')

        # True position
        ax.scatter(pts['x_true'], pts['y_true'], c='black',
                   marker='X', s=200, zorder=10, linewidths=1.5)

    # Add a single legend entry for the true position marker
    ax.scatter([], [], c='black', marker='X', s=120, label='True position')

    ax.set_xlabel('x [m]', fontsize=13)
    ax.set_ylabel('y [m]', fontsize=13)
    ax.set_title(f'Multi-User Position Scatter ({K} users, SNR={snr_db} dB)',
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
              shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-100, 100])
    ax.set_ylim([0, 200])
    fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # Build 3-UE system
    mu_system = MultiUserISACSystem(
        BW=30.72e6,
        ue_distances=UE_DISTANCES,
        ue_angles=UE_ANGLES_RAD,
        ue_zc_roots=UE_ZC_ROOTS,
    )

    print("=" * 60)
    print("  Multi-User Position Scatter")
    print("=" * 60)
    for k in range(mu_system.K):
        ue = mu_system.per_ue[k]
        x_t = ue['d0'] * np.cos(ue['theta'])
        y_t = ue['d0'] * np.sin(ue['theta'])
        print(f"  UE{k+1}: root={ue['zc_root']}, "
              f"d={ue['d0']:.0f}m, angle={ue['theta']*180/np.pi:.0f}deg "
              f"-> true pos=({x_t:.1f}, {y_t:.1f})m")

    # Run at SNR=20dB with 100 trials
    points = mu_position_scatter(mu_system, snr_db=20, num_trials=500,
                                  n_sequences=5)

    # Print per-UE stats
    for k in range(mu_system.K):
        pts = points[k]
        x_arr = np.array(pts['x_est'])
        y_arr = np.array(pts['y_est'])
        mean_x = np.mean(x_arr)
        mean_y = np.mean(y_arr)
        mean_err = np.sqrt((mean_x - pts['x_true'])**2 +
                           (mean_y - pts['y_true'])**2)
        print(f"  UE{k+1}: {len(x_arr)} valid estimates, "
              f"mean=({mean_x:.1f}, {mean_y:.1f})m, "
              f"true=({pts['x_true']:.1f}, {pts['y_true']:.1f})m, "
              f"mean error={mean_err:.1f}m")

    plot_scatter(points, mu_system.K, snr_db=20)
