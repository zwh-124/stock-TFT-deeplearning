import os
import numpy as np
import pandas as pd
import torch

import config
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser
from env import AShareTradingEnv
from plot import (plot_episode_return_hist, plot_avg_episode_trajectory,
                  plot_strategy_vs_benchmark, plot_excess_return_hist,
                  plot_intra_episode_mdd_hist)
from rl_utils import build_port_state, ObsCache


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
                         dropout=config.DROPOUT,
                         static_categorical=STATIC_CATEGORICAL,
                         static_n_continuous=len(STATIC_CONTINUOUS)).to(device)
    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=config.N_EXTRA_STATE, dropout=config.DROPOUT).to(device)

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
def run_one_episode(encoder, policy, env, obs_cache, device, start_idx, bins):
    """Run one 10-day episode from start_idx (fresh 1M cash, no randomization).

    Returns dict with nav_seq (nav_seq[0] = entry NAV ~= INIT_CAPITAL) and dates.
    """
    env.reset(start_date_idx=start_idx, episode_len=config.EPISODE_LEN)
    nav_seq = [env._compute_nav()]
    dates = [env.dates[start_idx]]
    for _ in range(config.EPISODE_LEN):
        date_idx = env.current_idx
        if date_idx >= len(env.dates):
            break
        date = env.dates[date_idx]
        done = False
        info = None
        for _phase in ["open", "close"]:
            dyn_t, stat_t, mask_t = obs_cache.get_obs(date_idx, env, device)
            port_state = build_port_state(env, device)
            enc = encoder(dyn_t, stat_t)
            dist = policy(enc, port_state, mask_t, env.phase)
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

        nav_seq.append(info['nav'] if info is not None else nav_seq[-1])
        dates.append(date)
        if done:
            break
    return {'start_idx': start_idx, 'start_date': dates[0],
            'end_date': dates[-1], 'nav_seq': nav_seq, 'dates': dates}


def get_overlap_starts(env):
    """Overlap sliding-window starts: every test trading day spawns a fresh
    EPISODE_LEN-day episode. Windows running past data end are dropped."""
    seq_len = config.SEQ_LEN
    test_dates = [i for i, d in enumerate(env.dates)
                  if config.TEST_START <= d <= config.TEST_END and i >= seq_len]
    n_dates = len(env.dates)
    starts = [d for d in test_dates if d + config.EPISODE_LEN - 1 < n_dates]
    dropped = len(test_dates) - len(starts)
    if dropped > 0:
        print(f"  Dropped {dropped} tail window(s) with < {config.EPISODE_LEN} "
              f"days remaining.")
    return starts


@torch.no_grad()
def run_backtest(encoder, policy, env, obs_cache, device):
    """Overlap-only episode backtest. Returns list of per-episode dicts."""
    bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)
    starts = get_overlap_starts(env)
    if not starts:
        print("No valid test windows found.")
        return None

    episodes = []
    total = len(starts)
    print(f"  Running {total} overlapping {config.EPISODE_LEN}-day episodes...")
    for k, start_idx in enumerate(starts):
        ep = run_one_episode(encoder, policy, env, obs_cache, device,
                             start_idx, bins)
        episodes.append(ep)
        if k % 20 == 0:
            r = ep['nav_seq'][-1] / ep['nav_seq'][0] - 1
            print(f"  [{k}/{total}] {ep['start_date']}->{ep['end_date']} "
                  f"r_e={r*100:+.2f}%")
    return episodes


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


def _intra_mdd(nav_seq):
    """Max drawdown within a single episode (cummax resets per episode)."""
    nav = np.asarray(nav_seq, dtype=np.float64)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def load_benchmark_close_map():
    """Return {trade_date(str) -> close} for CSI300, or None if unavailable."""
    path = os.path.join(config.MARKET_DIR, "000300.SH.csv")
    if not os.path.exists(path):
        return None
    bench = pd.read_csv(path)
    bench['trade_date'] = bench['trade_date'].astype(str)
    return dict(zip(bench['trade_date'], bench['close']))


def benchmark_window_return(close_map, start_date, end_date):
    """Window close-to-close return of benchmark over [start_date, end_date].

    Aligned per-episode with the strategy window. Note: the strategy also
    trades on start_date, so a ~half-day basis offset exists (acceptable
    approximation, both anchored to the same trading days)."""
    if close_map is None:
        return np.nan
    s, e = str(start_date), str(end_date)
    if s not in close_map or e not in close_map:
        return np.nan
    return close_map[e] / close_map[s] - 1.0


def compute_episode_metrics(episodes, close_map):
    """Overlap-episode distribution metrics. Returns (metrics, r, b, x, mdd)."""
    r = np.array([ep['nav_seq'][-1] / ep['nav_seq'][0] - 1.0
                  for ep in episodes], dtype=np.float64)
    b = np.array([benchmark_window_return(close_map, ep['start_date'],
                                          ep['end_date'])
                  for ep in episodes], dtype=np.float64)
    x = r - b
    mdd = np.array([_intra_mdd(ep['nav_seq']) for ep in episodes],
                   dtype=np.float64)
    E = len(r)
    std_r = r.std(ddof=1) if E > 1 else 0.0
    std_x = np.nanstd(x, ddof=1) if E > 1 else 0.0
    E_eff = max(E / config.EPISODE_LEN, 1.0)
    metrics = {
        # --- A: adapted from the 5 legacy metrics ---
        'mean_ret': float(r.mean()),
        'median_ret': float(np.median(r)),
        'std_ret': float(std_r),
        'episode_ir': float(r.mean() / (std_r + 1e-12)),
        'win_rate': float((r > 0).mean()),
        'avg_intra_mdd': float(mdd.mean()),
        'worst_intra_mdd': float(mdd.min()),
        # --- B: distribution / tail / excess ---
        'beat_bench_rate': float(np.nanmean(r > b)),
        'excess_mean': float(np.nanmean(x)),
        'excess_ir': float(np.nanmean(x) / (std_x + 1e-12)),
        'worst_ret': float(r.min()),
        'var5': float(np.quantile(r, 0.05)),
        'cond_loss': float(r[r < 0].mean()) if (r < 0).any() else 0.0,
        'se_mean': float(std_r / np.sqrt(E)) if E > 0 else 0.0,
        'se_mean_eff': float(std_r / np.sqrt(E_eff)),
        'n_episodes': E,
        'n_eff': float(E_eff),
    }
    return metrics, r, b, x, mdd


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

    print("Running RL backtest (overlap sliding-window episodes)...")
    grouped = df.sort_values('trade_date').groupby('ts_code')
    obs_cache = ObsCache(grouped, avail_features, env, config.SEQ_LEN)
    episodes = run_backtest(encoder, policy, env, obs_cache, device)
    if not episodes:
        print("Backtest produced no results.")
        return

    close_map = load_benchmark_close_map()
    m, r, b, x, mdd = compute_episode_metrics(episodes, close_map)

    print("\n=== RL 10-Day Episode Backtest (overlap) ===")
    print(f"Episodes (windows):   {m['n_episodes']}  "
          f"(effective n ~= {m['n_eff']:.1f})")
    print("[Return]")
    print(f"  Mean episode return r̄ : {m['mean_ret']*100:+.2f}%   "
          f"(median {m['median_ret']*100:+.2f}%, std {m['std_ret']*100:.2f}%)")
    print(f"  SE(mean):              {m['se_mean']*100:.3f}% "
          f"(overlapping, underestimated; eff-SE {m['se_mean_eff']*100:.3f}%)")
    print("[Win rate]")
    print(f"  P(r_e > 0):            {m['win_rate']*100:.1f}%")
    print(f"  P(r_e > benchmark):    {m['beat_bench_rate']*100:.1f}%")
    print("[Risk-adjusted]")
    print(f"  Episode IR (r̄/σ):      {m['episode_ir']:.3f}")
    print(f"  Excess IR (x̄/σ_x):     {m['excess_ir']:.3f}")
    print("[Downside / tail]")
    print(f"  Worst episode:         {m['worst_ret']*100:+.2f}%")
    print(f"  5% VaR:                {m['var5']*100:+.2f}%")
    print(f"  Conditional loss:      {m['cond_loss']*100:+.2f}%")
    print(f"  Avg intra-episode MDD: {m['avg_intra_mdd']*100:.2f}%")
    print(f"  Worst intra-ep MDD:    {m['worst_intra_mdd']*100:.2f}%")
    print("[Benchmark]")
    print(f"  Mean excess x̄:         {m['excess_mean']*100:+.2f}%")

    rows = [{'start_date': ep['start_date'], 'end_date': ep['end_date'],
             'r_e': r[i], 'b_e': b[i], 'x_e': x[i], 'intra_mdd': mdd[i]}
            for i, ep in enumerate(episodes)]
    ep_df = pd.DataFrame(rows)
    ep_path = os.path.join(config.CACHE_DIR, "backtest_rl_episodes.csv")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    ep_df.to_csv(ep_path, index=False)
    print(f"\nEpisode records saved to {ep_path}")

    plot_episode_return_hist(r, b, m, filename='rl_episode_return_hist.png')
    plot_avg_episode_trajectory(episodes, filename='rl_avg_trajectory.png')
    plot_strategy_vs_benchmark(episodes, r, b, filename='rl_vs_benchmark.png')
    plot_excess_return_hist(x, m, filename='rl_excess_hist.png')
    plot_intra_episode_mdd_hist(mdd, m, filename='rl_intra_mdd_hist.png')


if __name__ == "__main__":
    main()
