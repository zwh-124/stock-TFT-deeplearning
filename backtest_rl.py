import os
import numpy as np
import pandas as pd
import torch

import config
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser
from env import AShareTradingEnv
from plot import plot_backtest_nav
from rl_utils import get_obs_for_date, build_port_state, ObsCache


def load_rl_model(device, avail_features):
    ckpt_path = os.path.join(config.CACHE_DIR, "best_rl_policy.pt")
    if not os.path.exists(ckpt_path):
        print("No RL model found. Run train_rl.py first.")
        return None, None

    dynamic_dim = len(avail_features)
    static_dim = len(STATIC_FEATURES)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'config' in ckpt:
        saved_cfg = ckpt['config']
        for key, cur in {'HIDDEN_DIM': config.HIDDEN_DIM,
                         'SEQ_LEN': config.SEQ_LEN,
                         'NUM_HEADS': config.NUM_HEADS,
                         'N_BINS': config.N_BINS,
                         'N_HOLD': config.N_HOLD}.items():
            saved = saved_cfg.get(key)
            if saved is not None and saved != cur:
                print(f"ERROR: config mismatch — {key}: "
                      f"trained={saved}, current={cur}")
                return None, None

        saved_bins = saved_cfg.get('BINS')
        if saved_bins is not None and saved_bins != config.BINS:
            print(f"ERROR: BINS mismatch — "
                  f"trained={saved_bins}, current={config.BINS}")
            return None, None

        saved_features = saved_cfg.get('DYNAMIC_FEATURES')
        if saved_features is not None and saved_features != avail_features:
            missing = set(saved_features) - set(avail_features)
            extra = set(avail_features) - set(saved_features)
            print(f"ERROR: feature mismatch.\n"
                  f"  Missing from current: {missing}\n"
                  f"  Extra in current: {extra}")
            return None, None

    encoder = TFTEncoder(dynamic_dim, static_dim,
                         hidden_dim=config.HIDDEN_DIM,
                         seq_len=config.SEQ_LEN,
                         num_heads=config.NUM_HEADS,
                         dropout=config.DROPOUT).to(device)
    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=4, dropout=config.DROPOUT).to(device)

    if any(k.startswith('denoiser.') for k in ckpt["encoder"].keys()):
        denoiser = DiffusionDenoiser(
            feature_dim=dynamic_dim,
            seq_len=config.SEQ_LEN,
            hidden_dim=config.DIFFUSION_HIDDEN_DIM,
            time_dim=config.DIFFUSION_TIME_DIM,
            n_timesteps=config.DIFFUSION_T,
            beta_start=config.DIFFUSION_BETA_START,
            beta_end=config.DIFFUSION_BETA_END,
        ).to(device)
        encoder.denoiser = denoiser

    encoder.load_state_dict(ckpt["encoder"])
    policy.load_state_dict(ckpt["policy"])

    encoder.eval()
    policy.eval()
    return encoder, policy


def compute_metrics(nav_df):
    if isinstance(nav_df, list):
        nav_df = pd.DataFrame(nav_df)
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


@torch.no_grad()
def run_backtest(encoder, policy, env, obs_cache, device):
    seq_len = config.SEQ_LEN
    bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)

    test_dates = [i for i, d in enumerate(env.dates)
                  if config.TEST_START <= d <= config.TEST_END]
    test_dates = [i for i in test_dates if i >= seq_len]

    if not test_dates:
        print("No valid test dates found.")
        return None

    env.reset(start_date_idx=test_dates[0])
    nav_history = []

    for step, date_idx in enumerate(test_dates):
        date = env.dates[date_idx]
        env.current_idx = date_idx

        done = False
        for phase in ["open", "close"]:
            env.phase = phase
            dyn_t, stat_t, mask_t = obs_cache.get_obs(date_idx, env, device)
            port_state = build_port_state(env, device)

            enc = encoder(dyn_t, stat_t)
            dist = policy(enc, port_state, mask_t, phase)
            action = dist.probs.argmax(dim=-1)

            weights = bins[action].cpu().numpy()
            top_k_idx = np.argsort(weights)[-config.N_HOLD:]
            target_w = np.zeros(env.n_stocks)
            for idx in top_k_idx:
                target_w[idx] = weights[idx]
            w_sum = target_w.sum()
            if w_sum > 0:
                target_w = target_w / w_sum

            state, reward, done, info = env.step(target_w)
            if done:
                break

        nav = info['nav']
        prev_nav = nav_history[-1]['nav'] if nav_history else config.INIT_CAPITAL
        day_ret = (nav / prev_nav) - 1 if prev_nav > 0 else 0.0
        nav_history.append({'date': date, 'nav': nav, 'return': day_ret})

        if step % 20 == 0:
            print(f"  [{step}/{len(test_dates)}] {date} NAV={nav:,.0f}")

        if done:
            break

    return pd.DataFrame(nav_history)


def load_benchmark(dates):
    path = os.path.join(config.MARKET_DIR, "000300.SH.csv")
    if not os.path.exists(path):
        return None
    bench = pd.read_csv(path)
    bench['trade_date'] = bench['trade_date'].astype(str)
    bench = bench[bench['trade_date'].isin(dates)]
    bench = bench.sort_values('trade_date').reset_index(drop=True)
    if len(bench) < 2:
        return None
    bench['return'] = bench['close'].pct_change().fillna(0)
    bench['nav'] = (1 + bench['return']).cumprod() * config.INIT_CAPITAL
    return bench


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    df = build_merged_dataset(config.TRAIN_START, config.TEST_END)
    df, avail_features = build_features(df)

    encoder, policy = load_rl_model(device, avail_features)
    if encoder is None:
        return

    env = AShareTradingEnv(df)
    print(f"Stocks: {env.n_stocks}, Dates: {len(env.dates)}")

    print("Running RL backtest...")
    grouped = df.sort_values('trade_date').groupby('ts_code')
    obs_cache = ObsCache(grouped, avail_features, env, config.SEQ_LEN)
    nav_df = run_backtest(encoder, policy, env, obs_cache, device)
    if nav_df is None or len(nav_df) == 0:
        print("Backtest produced no results.")
        return

    metrics = compute_metrics(nav_df)
    print("\n=== RL Backtest Results ===")
    print(f"Annual Return: {metrics['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio:  {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:  {metrics['max_drawdown']*100:.2f}%")
    print(f"Total Return:  {metrics['total_return']*100:.2f}%")
    print(f"Dir Accuracy:  {metrics['direction_accuracy']*100:.2f}%")
    print(f"Trading Days:  {metrics['n_days']}")

    bench = load_benchmark(nav_df['date'].tolist())
    if bench is not None:
        bench_ret = bench['nav'].iloc[-1] / bench['nav'].iloc[0] - 1
        print(f"\nBenchmark (CSI300) Return: {bench_ret*100:.2f}%")
        print(f"Excess Return: "
              f"{(metrics['total_return']-bench_ret)*100:.2f}%")

    nav_path = os.path.join(config.CACHE_DIR, "backtest_rl_nav.csv")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    nav_df.to_csv(nav_path, index=False)
    print(f"\nNAV saved to {nav_path}")

    bench_path = os.path.join(config.MARKET_DIR, "000300.SH.csv")
    plot_backtest_nav(nav_path,
                      bench_csv=bench_path if os.path.exists(bench_path) else None)


if __name__ == "__main__":
    main()
