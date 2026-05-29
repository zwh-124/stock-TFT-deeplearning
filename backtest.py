import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import CompetitionTFT, DiffusionDenoiser
from dataset import StockDataset
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from plot import plot_backtest_nav
import config


@torch.no_grad()
def generate_predictions(model, dataset, device, close_idx):
    model.eval()
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    all_preds = []
    for x_dyn, x_stat, _ in loader:
        x_dyn = x_dyn.to(device)
        x_stat = x_stat.to(device)
        pred, _ = model(x_dyn, x_stat)
        all_preds.append(pred[:, close_idx].cpu().numpy())
    all_preds = np.concatenate(all_preds)
    records = []
    for i, s in enumerate(dataset.samples):
        records.append({
            'date': s['date'],
            'code': s['code'],
            'pred_score': all_preds[i],
            'y_true': s['y'][close_idx],
        })
    return pd.DataFrame(records)


def backtest_topk(pred_df, daily_df, n_hold=config.N_HOLD,
                  k_swap=config.K_SWAP, init_capital=config.INIT_CAPITAL):
    dates = sorted(pred_df['date'].unique())
    close_map = daily_df.set_index(['ts_code', 'trade_date'])['close'].to_dict()

    portfolio = []
    capital = init_capital
    nav_history = []

    for i, date in enumerate(dates):
        day_pred = pred_df[pred_df['date'] == date].sort_values(
            'pred_score', ascending=False)
        top_codes = day_pred['code'].head(n_hold).tolist()

        if i == 0:
            portfolio = top_codes[:n_hold]
        else:
            ranked = day_pred.set_index('code')['pred_score']
            port_scores = [(c, ranked.get(c, -999)) for c in portfolio]
            port_scores.sort(key=lambda x: x[1])
            sell_codes = [c for c, _ in port_scores[:k_swap]]
            buy_codes = [c for c in top_codes if c not in portfolio][:k_swap]
            portfolio = [c for c in portfolio if c not in sell_codes] + buy_codes

        day_returns = []
        for code in portfolio:
            next_date = dates[i + 1] if i + 1 < len(dates) else None
            if next_date is None:
                continue
            c_today = close_map.get((code, date))
            c_next = close_map.get((code, next_date))
            if c_today and c_next and c_today > 0:
                day_returns.append((c_next - c_today) / c_today)

        if day_returns:
            port_ret = np.mean(day_returns)
        else:
            port_ret = 0.0
        capital *= (1 + port_ret)
        nav_history.append({'date': date, 'nav': capital, 'return': port_ret})

    nav_df = pd.DataFrame(nav_history)
    return nav_df


def compute_metrics(nav_df):
    returns = nav_df['return'].values
    n_days = len(returns)
    total_return = nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    sharpe = np.mean(returns) / (np.std(returns, ddof=1) + 1e-8) * np.sqrt(252)
    cummax = nav_df['nav'].cummax()
    drawdown = (nav_df['nav'] - cummax) / cummax
    max_drawdown = drawdown.min()
    dir_acc = (returns[1:] > 0).sum() / max(len(returns) - 1, 1)
    return {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'n_days': n_days,
        'direction_accuracy': dir_acc,
    }


def load_benchmark(benchmark_code='000300.SH', dates=None):
    path = os.path.join(config.MARKET_DIR, f"{benchmark_code}.csv")
    bench = pd.read_csv(path)
    bench['trade_date'] = bench['trade_date'].astype(str)
    if dates is not None:
        bench = bench[bench['trade_date'].isin(dates)]
    bench = bench.sort_values('trade_date').reset_index(drop=True)
    bench['return'] = bench['close'].pct_change().fillna(0)
    bench['nav'] = (1 + bench['return']).cumprod() * config.INIT_CAPITAL
    return bench


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config.CACHE_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("No trained model found. Run train.py first.")
        return

    print("Loading data...")
    df = build_merged_dataset(start_date=config.TEST_START,
                              end_date=config.TEST_END)
    df, avail_features = build_features(df)

    val_ds = StockDataset(df, avail_features, STATIC_FEATURES,
                          cache_tag="test")
    print(f"Test samples: {len(val_ds)}")

    model = CompetitionTFT(
        dynamic_input_dim=len(avail_features),
        static_input_dim=len(STATIC_FEATURES),
        hidden_dim=config.HIDDEN_DIM,
        seq_len=config.SEQ_LEN,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
        static_categorical=STATIC_CATEGORICAL,
        static_n_continuous=len(STATIC_CONTINUOUS),
        avail_features=avail_features,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'config' in ckpt:
        saved_cfg = ckpt['config']
        for key, cur in {'USE_CSI300': config.USE_CSI300,
                         'HIDDEN_DIM': config.HIDDEN_DIM,
                         'SEQ_LEN': config.SEQ_LEN,
                         'NUM_HEADS': config.NUM_HEADS,
                         'PRED_HORIZON': config.PRED_HORIZON}.items():
            saved = saved_cfg.get(key)
            if saved is not None and saved != cur:
                print(f"WARNING: config mismatch — {key}: "
                      f"trained={saved}, current={cur}")
        sd = ckpt['state_dict']
    else:
        sd = ckpt

    if any(k.startswith('encoder.denoiser.') for k in sd.keys()):
        denoiser = DiffusionDenoiser(
            feature_dim=len(avail_features),
            seq_len=config.SEQ_LEN,
            hidden_dim=config.DIFFUSION_HIDDEN_DIM,
            time_dim=config.DIFFUSION_TIME_DIM,
            n_timesteps=config.DIFFUSION_T,
            beta_start=config.DIFFUSION_BETA_START,
            beta_end=config.DIFFUSION_BETA_END,
        ).to(device)
        model.encoder.denoiser = denoiser

    model.load_state_dict(sd)

    close_idx = avail_features.index('close')
    print("Generating predictions...")
    pred_df = generate_predictions(model, val_ds, device, close_idx)

    daily_df = df[['ts_code', 'trade_date', 'close']].copy()
    daily_df.rename(columns={'trade_date': 'trade_date'}, inplace=True)

    print("Running backtest...")
    nav_df = backtest_topk(pred_df, daily_df)
    metrics = compute_metrics(nav_df)

    print("\n=== Backtest Results ===")
    print(f"Annual Return: {metrics['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio:  {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:  {metrics['max_drawdown']*100:.2f}%")
    print(f"Total Return:  {metrics['total_return']*100:.2f}%")
    print(f"Dir Accuracy:  {metrics['direction_accuracy']*100:.2f}%")
    print(f"Trading Days:  {metrics['n_days']}")

    bench = load_benchmark('000300.SH', nav_df['date'].tolist())
    if len(bench) > 1:
        bench_ret = bench['nav'].iloc[-1] / bench['nav'].iloc[0] - 1
        print(f"\nBenchmark (CSI300) Return: {bench_ret*100:.2f}%")
        print(f"Excess Return: {(metrics['total_return']-bench_ret)*100:.2f}%")

    nav_path = os.path.join(config.CACHE_DIR, "backtest_nav.csv")
    nav_df.to_csv(nav_path, index=False)
    print(f"\nNAV saved to {nav_path}")

    bench_path = os.path.join(config.MARKET_DIR, "000300.SH.csv")
    plot_backtest_nav(nav_path,
                      bench_csv=bench_path if os.path.exists(bench_path) else None)


if __name__ == "__main__":
    main()
