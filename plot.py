import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def plot_training_curves(history, save_dir=FIGURES_DIR):
    """Plot train/val loss, IC, and ICIR curves."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(epochs, history['ic'], marker='o', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IC')
    axes[1].set_title('IC')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    axes[2].plot(epochs, history['icir'], marker='o', markersize=3)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('ICIR')
    axes[2].set_title('ICIR')
    axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved to {path}")


def plot_ic_distribution(daily_ics, save_dir=FIGURES_DIR):
    """Plot histogram of daily IC values."""
    os.makedirs(save_dir, exist_ok=True)
    ics = np.array(daily_ics)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ics, bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(x=ics.mean(), color='red', linestyle='--',
               label=f'Mean={ics.mean():.4f}')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('IC')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Daily IC Distribution (n={len(ics)}, '
                 f'IC={ics.mean():.4f}, ICIR={ics.mean()/(ics.std()+1e-8):.4f})')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'ic_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"IC distribution saved to {path}")


def plot_backtest_nav(nav_csv, bench_csv=None, save_dir=FIGURES_DIR,
                     filename='backtest_nav.png'):
    os.makedirs(save_dir, exist_ok=True)
    nav_df = pd.read_csv(nav_csv)
    nav_df['date'] = pd.to_datetime(nav_df['date'], format='%Y%m%d')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(nav_df['date'], nav_df['nav'], label='Strategy', linewidth=1.5)

    if bench_csv and os.path.exists(bench_csv):
        bench = pd.read_csv(bench_csv)
        bench['trade_date'] = pd.to_datetime(
            bench['trade_date'].astype(str), format='%Y%m%d')
        bench = bench[bench['trade_date'].isin(nav_df['date'])]
        bench = bench.sort_values('trade_date').reset_index(drop=True)
        if len(bench) > 1:
            init_nav = nav_df['nav'].iloc[0]
            bench_nav = bench['close'] / bench['close'].iloc[0] * init_nav
            ax.plot(bench['trade_date'], bench_nav,
                    label='CSI300', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('NAV')
    ax.set_title('Backtest NAV Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Backtest NAV curve saved to {path}")


def plot_rl_reward_curve(reward_history, save_dir=FIGURES_DIR):
    """Plot RL training reward curve with rolling average."""
    os.makedirs(save_dir, exist_ok=True)
    rewards = np.array(reward_history)
    episodes = range(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.8,
            label='Per-episode')
    window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
    if window > 1:
        rolling = pd.Series(rewards).rolling(window).mean().values
        ax.plot(episodes, rolling, color='red', linewidth=1.5,
                label=f'Rolling {window}')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Reward')
    ax.set_title('RL Training Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'rl_reward_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"RL reward curve saved to {path}")


def plot_greedy_eval_curve(eval_history, save_dir=FIGURES_DIR):
    """Plot the fixed-window greedy-eval learning curve.

    eval_history: list of (step, alpha, mean_ret, episode_ir, win_rate).
    alpha (excess vs CSI300) is the model-selection metric; this is the real
    learning curve, unlike the per-step training reward.
    """
    if not eval_history:
        return
    os.makedirs(save_dir, exist_ok=True)
    arr = np.array(eval_history, dtype=np.float64)
    steps = arr[:, 0]
    alpha, mean_ret, ep_ir, win = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, alpha * 100, marker='o', ms=3, color='teal')
    axes[0].axhline(0, color='gray', ls='--', lw=0.6)
    best_i = int(np.argmax(alpha))
    axes[0].plot(steps[best_i], alpha[best_i] * 100, marker='*', ms=14,
                 color='red', label=f'best={alpha[best_i]*100:+.2f}%')
    axes[0].set_xlabel('RL step')
    axes[0].set_ylabel('Alpha vs CSI300 (%)')
    axes[0].set_title('Greedy-Eval Alpha (selection metric)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mean_ret * 100, marker='o', ms=3, color='steelblue')
    axes[1].axhline(0, color='gray', ls='--', lw=0.6)
    axes[1].set_xlabel('RL step')
    axes[1].set_ylabel('Mean episode return (%)')
    axes[1].set_title('Greedy-Eval Return')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, ep_ir, marker='o', ms=3, color='darkorange',
                 label='Episode IR')
    axes[2].plot(steps, win, marker='s', ms=3, color='green',
                 label='Win rate')
    axes[2].axhline(0, color='gray', ls='--', lw=0.6)
    axes[2].set_xlabel('RL step')
    axes[2].set_title('Greedy-Eval IR / Win rate')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'rl_greedy_eval_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Greedy-eval curve saved to {path}")


def plot_drawdown(nav_csv, save_dir=FIGURES_DIR, filename='drawdown.png'):
    """Plot drawdown curve from NAV history."""
    os.makedirs(save_dir, exist_ok=True)
    nav_df = pd.read_csv(nav_csv)
    nav_df['date'] = pd.to_datetime(nav_df['date'], format='%Y%m%d')

    nav = nav_df['nav'].values
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(nav_df['date'], drawdown, 0, alpha=0.4, color='red')
    ax.plot(nav_df['date'], drawdown, color='darkred', linewidth=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(f'Drawdown (Max: {drawdown.min():.2f}%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Drawdown curve saved to {path}")


def plot_feature_importance(gate_weights, feature_names, save_dir=FIGURES_DIR):
    """Plot feature importance from learned gate weights."""
    os.makedirs(save_dir, exist_ok=True)
    weights = np.array(gate_weights)
    if weights.ndim == 2:
        weights = weights.mean(axis=0)

    sorted_idx = np.argsort(weights)[::-1]
    top_n = min(20, len(weights))
    idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [feature_names[i] for i in idx]
    vals = weights[idx]
    ax.barh(range(top_n), vals[::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel('Gate Weight')
    ax.set_title('Feature Importance (Top 20)')
    plt.tight_layout()
    path = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature importance saved to {path}")


def plot_holdings_heatmap(nav_df, holdings_history, codes,
                          save_dir=FIGURES_DIR, filename='holdings_heatmap.png'):
    """Plot daily holdings heatmap (top stocks only)."""
    os.makedirs(save_dir, exist_ok=True)
    holdings = np.array(holdings_history)
    if holdings.ndim != 2:
        print("Holdings history format invalid, skipping heatmap.")
        return

    total_per_stock = np.abs(holdings).sum(axis=0)
    top_idx = np.argsort(total_per_stock)[-20:]
    top_holdings = holdings[:, top_idx]
    top_codes = [codes[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(top_holdings.T, aspect='auto', cmap='RdYlGn',
                   interpolation='nearest')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Stock')
    ax.set_yticks(range(len(top_codes)))
    ax.set_yticklabels(top_codes, fontsize=7)
    ax.set_title('Daily Holdings Heatmap (Top 20 Stocks)')
    plt.colorbar(im, ax=ax, label='Weight')
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Holdings heatmap saved to {path}")


def plot_episode_return_hist(r, b, metrics, save_dir=FIGURES_DIR,
                             filename='rl_episode_return_hist.png'):
    """Histogram of 10-day episode returns r_e, overlaid with benchmark b_e."""
    os.makedirs(save_dir, exist_ok=True)
    r = np.asarray(r) * 100
    b = np.asarray(b) * 100
    b = b[~np.isnan(b)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(r, bins=40, alpha=0.6, color='steelblue',
            edgecolor='black', label='Strategy r_e')
    if len(b) > 0:
        ax.hist(b, bins=40, alpha=0.4, color='orange',
                edgecolor='none', label='Benchmark b_e')
    ax.axvline(metrics['mean_ret'] * 100, color='red', linestyle='--',
               label=f"Mean={metrics['mean_ret']*100:+.2f}%")
    ax.axvline(metrics['median_ret'] * 100, color='green', linestyle=':',
               label=f"Median={metrics['median_ret']*100:+.2f}%")
    ax.axvline(metrics['var5'] * 100, color='purple', linestyle='-.',
               label=f"VaR5={metrics['var5']*100:+.2f}%")
    ax.axvline(0, color='gray', linewidth=0.6)
    ax.set_xlabel('10-day episode return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Episode Return Distribution (E={metrics['n_episodes']}, "
                 f"WinRate={metrics['win_rate']*100:.1f}%, "
                 f"IR={metrics['episode_ir']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Episode return histogram saved to {path}")


def plot_avg_episode_trajectory(episodes, save_dir=FIGURES_DIR,
                                filename='rl_avg_trajectory.png'):
    """Average NAV trajectory (day 0..EPISODE_LEN), all episodes rebased to 1.0."""
    os.makedirs(save_dir, exist_ok=True)
    max_len = max(len(ep['nav_seq']) for ep in episodes)
    rebased = np.full((len(episodes), max_len), np.nan)
    for i, ep in enumerate(episodes):
        seq = np.asarray(ep['nav_seq'], dtype=np.float64)
        if seq[0] > 0:
            rebased[i, :len(seq)] = seq / seq[0]

    days = np.arange(max_len)
    mean_traj = np.nanmean(rebased, axis=0)
    std_traj = np.nanstd(rebased, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(days, mean_traj, color='steelblue', linewidth=2, label='Mean trajectory')
    ax.fill_between(days, mean_traj - std_traj, mean_traj + std_traj,
                    alpha=0.25, color='steelblue', label='±1σ')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.6)
    ax.set_xlabel('Day within episode')
    ax.set_ylabel('NAV (rebased to 1.0 at entry)')
    ax.set_title(f'Average 10-Day Episode Trajectory (E={len(episodes)})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Average episode trajectory saved to {path}")


def plot_strategy_vs_benchmark(episodes, r, b, save_dir=FIGURES_DIR,
                               filename='rl_vs_benchmark.png'):
    """Per-episode paired bars: strategy r_e vs benchmark b_e, win=green/loss=red."""
    os.makedirs(save_dir, exist_ok=True)
    r = np.asarray(r) * 100
    b = np.asarray(b) * 100
    dates = [pd.to_datetime(str(ep['start_date']), format='%Y%m%d')
             for ep in episodes]
    xpos = np.arange(len(episodes))
    colors = ['green' if r[i] > b[i] else 'red' for i in range(len(r))]

    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.42
    ax.bar(xpos - width / 2, r, width, color=colors, alpha=0.8,
           label='Strategy r_e (green=beat bench)')
    ax.bar(xpos + width / 2, b, width, color='gray', alpha=0.5,
           label='Benchmark b_e')
    ax.axhline(0, color='black', linewidth=0.6)
    n_ticks = min(12, len(episodes))
    tick_idx = np.linspace(0, len(episodes) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([dates[i].strftime('%y-%m-%d') for i in tick_idx],
                       rotation=45, fontsize=8)
    ax.set_xlabel('Episode start date')
    ax.set_ylabel('10-day return (%)')
    ax.set_title('Strategy vs Benchmark by Episode Window')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Strategy-vs-benchmark plot saved to {path}")


def plot_excess_return_hist(x, metrics, save_dir=FIGURES_DIR,
                            filename='rl_excess_hist.png'):
    """Histogram of per-episode excess return x_e = r_e - b_e."""
    os.makedirs(save_dir, exist_ok=True)
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=40, alpha=0.7, color='teal', edgecolor='black')
    ax.axvline(metrics['excess_mean'] * 100, color='red', linestyle='--',
               label=f"Mean excess={metrics['excess_mean']*100:+.2f}%")
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Excess return x_e = r_e - b_e (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Excess Return Distribution "
                 f"(BeatRate={metrics['beat_bench_rate']*100:.1f}%, "
                 f"ExcessIR={metrics['excess_ir']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Excess return histogram saved to {path}")


def plot_intra_episode_mdd_hist(intra_mdd, metrics, save_dir=FIGURES_DIR,
                                filename='rl_intra_mdd_hist.png'):
    """Histogram of per-episode intra-window max drawdown."""
    os.makedirs(save_dir, exist_ok=True)
    mdd = np.asarray(intra_mdd, dtype=np.float64) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(mdd, bins=40, alpha=0.7, color='indianred', edgecolor='black')
    ax.axvline(metrics['avg_intra_mdd'] * 100, color='blue', linestyle='--',
               label=f"Avg MDD={metrics['avg_intra_mdd']*100:.2f}%")
    ax.axvline(metrics['worst_intra_mdd'] * 100, color='black', linestyle='-.',
               label=f"Worst MDD={metrics['worst_intra_mdd']*100:.2f}%")
    ax.set_xlabel('Intra-episode max drawdown (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Intra-Episode Max Drawdown Distribution "
                 f"(E={metrics['n_episodes']})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Intra-episode MDD histogram saved to {path}")
