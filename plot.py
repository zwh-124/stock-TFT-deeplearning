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
