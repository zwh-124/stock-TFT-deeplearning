import os
import pandas as pd
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


def plot_backtest_nav(nav_csv, bench_csv=None, save_dir=FIGURES_DIR):
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
    path = os.path.join(save_dir, 'backtest_nav.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Backtest NAV curve saved to {path}")
